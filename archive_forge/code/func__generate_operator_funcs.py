import types
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
def _generate_operator_funcs(self, yaql_operators, engine):
    binary_doc = ''
    unary_doc = ''
    precedence_dict = {}
    for up, bp, op_name, op_alias in yaql_operators.operators.values():
        self._aliases[op_name] = op_alias
        if up:
            la = precedence_dict.setdefault((abs(up), 'l' if up > 0 else 'r'), [])
            la.append('UNARY_' + op_name if bp else op_name)
            unary_doc += 'value : ' if not unary_doc else '\n| '
            spec_prefix = '{0} value' if up > 0 else 'value {0}'
            if bp:
                unary_doc += (spec_prefix + ' %prec UNARY_{0}').format(op_name)
            else:
                unary_doc += spec_prefix.format(op_name)
        if bp:
            la = precedence_dict.setdefault((abs(bp), 'l' if bp > 0 else 'r'), [])
            if op_name == 'INDEXER':
                la.extend(('LIST', 'INDEXER'))
            elif op_name == 'MAP':
                la.append('MAP')
            else:
                la.append(op_name)
                binary_doc += ('value : ' if not binary_doc else '\n| ') + 'value {0} value'.format(op_name)

    def p_binary(this, p):
        alias = this._aliases.get(p.slice[2].type)
        p[0] = expressions.BinaryOperator(p[2], p[1], p[3], alias)

    def p_unary(this, p):
        if p[1] in yaql_operators.operators:
            alias = this._aliases.get(p.slice[1].type)
            p[0] = expressions.UnaryOperator(p[1], p[2], alias)
        else:
            alias = this._aliases.get(p.slice[2].type)
            p[0] = expressions.UnaryOperator(p[2], p[1], alias)
    p_binary.__doc__ = binary_doc
    self.p_binary = types.MethodType(p_binary, self)
    p_unary.__doc__ = unary_doc
    self.p_unary = types.MethodType(p_unary, self)
    precedence = []
    for i in range(1, len(precedence_dict) + 1):
        for oa in ('r', 'l'):
            value = precedence_dict.get((i, oa))
            if value:
                precedence.append((('left',) if oa == 'l' else ('right',)) + tuple(value))
    precedence.insert(0, ('left', ','))
    precedence.reverse()
    self.precedence = tuple(precedence)

    def p_value_call(this, p):
        """
            func : value '(' args ')'
            """
        arg = ()
        if len(p) > 4:
            arg = p[3]
        p[0] = expressions.Function('#call', p[1], *arg)
    if engine.allow_delegates:
        self.p_value_call = types.MethodType(p_value_call, self)