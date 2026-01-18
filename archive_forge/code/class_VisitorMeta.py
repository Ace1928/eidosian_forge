from . import ast
class VisitorMeta(type):

    def __new__(cls, name, bases, attrs):
        enter_handlers = {}
        leave_handlers = {}
        for base in bases:
            if hasattr(base, '_enter_handlers'):
                enter_handlers.update(base._enter_handlers)
            if hasattr(base, '_leave_handlers'):
                leave_handlers.update(base._leave_handlers)
        for attr, val in attrs.items():
            if attr.startswith('enter_'):
                ast_kind = attr[6:]
                ast_type = AST_KIND_TO_TYPE.get(ast_kind)
                enter_handlers[ast_type] = val
            elif attr.startswith('leave_'):
                ast_kind = attr[6:]
                ast_type = AST_KIND_TO_TYPE.get(ast_kind)
                leave_handlers[ast_type] = val
        attrs['_enter_handlers'] = enter_handlers
        attrs['_leave_handlers'] = leave_handlers
        attrs['_get_enter_handler'] = enter_handlers.get
        attrs['_get_leave_handler'] = leave_handlers.get
        return super(VisitorMeta, cls).__new__(cls, name, bases, attrs)