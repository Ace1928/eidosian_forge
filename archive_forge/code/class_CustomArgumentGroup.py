import argparse
class CustomArgumentGroup(argparse._ArgumentGroup):
    """
    Custom argument group that allows for the use of `-` or `_` in arguments passed and overrides the help for each
    when applicable.
    """

    def _add_action(self, action):
        args = vars(action)
        if isinstance(action, argparse._StoreTrueAction):
            action = _StoreTrueAction(args['option_strings'], args['dest'], args['default'], args['required'], args['help'])
        elif isinstance(action, argparse._StoreConstAction):
            action = _StoreConstAction(args['option_strings'], args['dest'], args['const'], args['default'], args['required'], args['help'])
        elif isinstance(action, argparse._StoreAction):
            action = _StoreAction(**args)
        action = super()._add_action(action)
        return action