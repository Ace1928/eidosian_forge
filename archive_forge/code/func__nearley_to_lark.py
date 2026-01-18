import os.path
import sys
import codecs
import argparse
from lark import Lark, Transformer, v_args
def _nearley_to_lark(g, builtin_path, n2l, js_code, folder_path, includes):
    rule_defs = []
    tree = nearley_grammar_parser.parse(g)
    for statement in tree.children:
        if statement.data == 'directive':
            directive, arg = statement.children
            if directive in ('builtin', 'include'):
                folder = builtin_path if directive == 'builtin' else folder_path
                path = os.path.join(folder, arg[1:-1])
                if path not in includes:
                    includes.add(path)
                    with codecs.open(path, encoding='utf8') as f:
                        text = f.read()
                    rule_defs += _nearley_to_lark(text, builtin_path, n2l, js_code, os.path.abspath(os.path.dirname(path)), includes)
            else:
                assert False, directive
        elif statement.data == 'js_code':
            code, = statement.children
            code = code[2:-2]
            js_code.append(code)
        elif statement.data == 'macro':
            pass
        elif statement.data == 'ruledef':
            rule_defs.append(n2l.transform(statement))
        else:
            raise Exception('Unknown statement: %s' % statement)
    return rule_defs