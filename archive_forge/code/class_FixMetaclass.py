from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, \
class FixMetaclass(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = '\n    classdef<any*>\n    '

    def transform(self, node, results):
        if not has_metaclass(node):
            return
        fixup_parse_tree(node)
        last_metaclass = None
        for suite, i, stmt in find_metas(node):
            last_metaclass = stmt
            stmt.remove()
        text_type = node.children[0].type
        if len(node.children) == 7:
            if node.children[3].type == syms.arglist:
                arglist = node.children[3]
            else:
                parent = node.children[3].clone()
                arglist = Node(syms.arglist, [parent])
                node.set_child(3, arglist)
        elif len(node.children) == 6:
            arglist = Node(syms.arglist, [])
            node.insert_child(3, arglist)
        elif len(node.children) == 4:
            arglist = Node(syms.arglist, [])
            node.insert_child(2, Leaf(token.RPAR, u')'))
            node.insert_child(2, arglist)
            node.insert_child(2, Leaf(token.LPAR, u'('))
        else:
            raise ValueError('Unexpected class definition')
        meta_txt = last_metaclass.children[0].children[0]
        meta_txt.value = 'metaclass'
        orig_meta_prefix = meta_txt.prefix
        touch_import(u'future.utils', u'with_metaclass', node)
        metaclass = last_metaclass.children[0].children[2].clone()
        metaclass.prefix = u''
        arguments = [metaclass]
        if arglist.children:
            if len(arglist.children) == 1:
                base = arglist.children[0].clone()
                base.prefix = u' '
            else:
                bases = parenthesize(arglist.clone())
                bases.prefix = u' '
                base = Call(Name('type'), [String("'NewBase'"), Comma(), bases, Comma(), Node(syms.atom, [Leaf(token.LBRACE, u'{'), Leaf(token.RBRACE, u'}')], prefix=u' ')], prefix=u' ')
            arguments.extend([Comma(), base])
        arglist.replace(Call(Name(u'with_metaclass', prefix=arglist.prefix), arguments))
        fixup_indent(suite)
        if not suite.children:
            suite.remove()
            pass_leaf = Leaf(text_type, u'pass')
            pass_leaf.prefix = orig_meta_prefix
            node.append_child(pass_leaf)
            node.append_child(Leaf(token.NEWLINE, u'\n'))
        elif len(suite.children) > 1 and (suite.children[-2].type == token.INDENT and suite.children[-1].type == token.DEDENT):
            pass_leaf = Leaf(text_type, u'pass')
            suite.insert_child(-1, pass_leaf)
            suite.insert_child(-1, Leaf(token.NEWLINE, u'\n'))