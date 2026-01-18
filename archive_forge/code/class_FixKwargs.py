from lib2to3 import fixer_base
from lib2to3.fixer_util import token, String, Newline, Comma, Name
from libfuturize.fixer_util import indentation, suitify, DoubleStar
class FixKwargs(fixer_base.BaseFix):
    run_order = 7
    PATTERN = u"funcdef< 'def' NAME parameters< '(' arglist=typedargslist< params=any* > ')' > ':' suite=any >"

    def transform(self, node, results):
        params_rawlist = results[u'params']
        for i, item in enumerate(params_rawlist):
            if item.type == token.STAR:
                params_rawlist = params_rawlist[i:]
                break
        else:
            return
        new_kwargs = needs_fixing(params_rawlist)
        if not new_kwargs:
            return
        suitify(node)
        suite = node.children[4]
        first_stmt = suite.children[2]
        ident = indentation(first_stmt)
        for name, default_value in gen_params(params_rawlist):
            if default_value is None:
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_assign_template % {u'name': name, u'kwargs': new_kwargs}, prefix=ident))
            else:
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_else_template % {u'name': name, u'default': default_value}, prefix=ident))
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_if_template % {u'assign': _assign_template % {u'name': name, u'kwargs': new_kwargs}, u'name': name, u'kwargs': new_kwargs}, prefix=ident))
        first_stmt.prefix = ident
        suite.children[2].prefix = u''
        must_add_kwargs = remove_params(params_rawlist)
        if must_add_kwargs:
            arglist = results[u'arglist']
            if len(arglist.children) > 0 and arglist.children[-1].type != token.COMMA:
                arglist.append_child(Comma())
            arglist.append_child(DoubleStar(prefix=u' '))
            arglist.append_child(Name(new_kwargs))