import functools
import re
import warnings
@BaseSpec.register_syntax
class NpmSpec(BaseSpec):
    SYNTAX = 'npm'

    @classmethod
    def _parse_to_clause(cls, expression):
        return cls.Parser.parse(expression)

    class Parser:
        JOINER = '||'
        HYPHEN = ' - '
        NUMBER = 'x|X|\\*|0|[1-9][0-9]*'
        PART = '[a-zA-Z0-9.-]*'
        NPM_SPEC_BLOCK = re.compile('\n            ^(?:v)?                     # Strip optional initial v\n            (?P<op><|<=|>=|>|=|\\^|~|)   # Operator, can be empty\n            (?P<major>{nb})(?:\\.(?P<minor>{nb})(?:\\.(?P<patch>{nb}))?)?\n            (?:-(?P<prerel>{part}))?    # Optional re-release\n            (?:\\+(?P<build>{part}))?    # Optional build\n            $'.format(nb=NUMBER, part=PART), re.VERBOSE)

        @classmethod
        def range(cls, operator, target):
            return Range(operator, target, prerelease_policy=Range.PRERELEASE_SAMEPATCH)

        @classmethod
        def parse(cls, expression):
            result = Never()
            groups = expression.split(cls.JOINER)
            for group in groups:
                group = group.strip()
                if not group:
                    group = '>=0.0.0'
                subclauses = []
                if cls.HYPHEN in group:
                    low, high = group.split(cls.HYPHEN, 2)
                    subclauses = cls.parse_simple('>=' + low) + cls.parse_simple('<=' + high)
                else:
                    blocks = group.split(' ')
                    for block in blocks:
                        if not cls.NPM_SPEC_BLOCK.match(block):
                            raise ValueError('Invalid NPM block in %r: %r' % (expression, block))
                        subclauses.extend(cls.parse_simple(block))
                prerelease_clauses = []
                non_prerel_clauses = []
                for clause in subclauses:
                    if clause.target.prerelease:
                        if clause.operator in (Range.OP_GT, Range.OP_GTE):
                            prerelease_clauses.append(Range(operator=Range.OP_LT, target=Version(major=clause.target.major, minor=clause.target.minor, patch=clause.target.patch + 1), prerelease_policy=Range.PRERELEASE_ALWAYS))
                        elif clause.operator in (Range.OP_LT, Range.OP_LTE):
                            prerelease_clauses.append(Range(operator=Range.OP_GTE, target=Version(major=clause.target.major, minor=clause.target.minor, patch=0, prerelease=()), prerelease_policy=Range.PRERELEASE_ALWAYS))
                        prerelease_clauses.append(clause)
                        non_prerel_clauses.append(cls.range(operator=clause.operator, target=clause.target.truncate()))
                    else:
                        non_prerel_clauses.append(clause)
                if prerelease_clauses:
                    result |= AllOf(*prerelease_clauses)
                result |= AllOf(*non_prerel_clauses)
            return result
        PREFIX_CARET = '^'
        PREFIX_TILDE = '~'
        PREFIX_EQ = '='
        PREFIX_GT = '>'
        PREFIX_GTE = '>='
        PREFIX_LT = '<'
        PREFIX_LTE = '<='
        PREFIX_ALIASES = {'': PREFIX_EQ}
        PREFIX_TO_OPERATOR = {PREFIX_EQ: Range.OP_EQ, PREFIX_LT: Range.OP_LT, PREFIX_LTE: Range.OP_LTE, PREFIX_GTE: Range.OP_GTE, PREFIX_GT: Range.OP_GT}
        EMPTY_VALUES = ['*', 'x', 'X', None]

        @classmethod
        def parse_simple(cls, simple):
            match = cls.NPM_SPEC_BLOCK.match(simple)
            prefix, major_t, minor_t, patch_t, prerel, build = match.groups()
            prefix = cls.PREFIX_ALIASES.get(prefix, prefix)
            major = None if major_t in cls.EMPTY_VALUES else int(major_t)
            minor = None if minor_t in cls.EMPTY_VALUES else int(minor_t)
            patch = None if patch_t in cls.EMPTY_VALUES else int(patch_t)
            if build is not None and prefix not in [cls.PREFIX_EQ]:
                build = None
            if major is None:
                target = Version(major=0, minor=0, patch=0)
                if prefix not in [cls.PREFIX_EQ, cls.PREFIX_GTE]:
                    raise ValueError('Invalid expression %r' % simple)
                prefix = cls.PREFIX_GTE
            elif minor is None:
                target = Version(major=major, minor=0, patch=0)
            elif patch is None:
                target = Version(major=major, minor=minor, patch=0)
            else:
                target = Version(major=major, minor=minor, patch=patch, prerelease=prerel.split('.') if prerel else (), build=build.split('.') if build else ())
            if (major is None or minor is None or patch is None) and (prerel or build):
                raise ValueError('Invalid NPM spec: %r' % simple)
            if prefix == cls.PREFIX_CARET:
                if target.major:
                    high = target.truncate().next_major()
                elif target.minor:
                    high = target.truncate().next_minor()
                elif minor is None:
                    high = target.truncate().next_major()
                elif patch is None:
                    high = target.truncate().next_minor()
                else:
                    high = target.truncate().next_patch()
                return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, high)]
            elif prefix == cls.PREFIX_TILDE:
                assert major is not None
                if minor is None:
                    high = target.next_major()
                else:
                    high = target.next_minor()
                return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, high)]
            elif prefix == cls.PREFIX_EQ:
                if major is None:
                    return [cls.range(Range.OP_GTE, target)]
                elif minor is None:
                    return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, target.next_major())]
                elif patch is None:
                    return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, target.next_minor())]
                else:
                    return [cls.range(Range.OP_EQ, target)]
            elif prefix == cls.PREFIX_GT:
                assert major is not None
                if minor is None:
                    return [cls.range(Range.OP_GTE, target.next_major())]
                elif patch is None:
                    return [cls.range(Range.OP_GTE, target.next_minor())]
                else:
                    return [cls.range(Range.OP_GT, target)]
            elif prefix == cls.PREFIX_GTE:
                return [cls.range(Range.OP_GTE, target)]
            elif prefix == cls.PREFIX_LT:
                assert major is not None
                return [cls.range(Range.OP_LT, target)]
            else:
                assert prefix == cls.PREFIX_LTE
                assert major is not None
                if minor is None:
                    return [cls.range(Range.OP_LT, target.next_major())]
                elif patch is None:
                    return [cls.range(Range.OP_LT, target.next_minor())]
                else:
                    return [cls.range(Range.OP_LTE, target)]