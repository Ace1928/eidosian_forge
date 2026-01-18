import argparse
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class ActionFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('name', 'Name'), ('is_system', 'Is system'), ('input', 'Input'), ('description', 'Description'), ('tags', 'Tags'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]

    @staticmethod
    def format(action=None, lister=False):
        if action:
            tags = getattr(action, 'tags', None) or []
            input_ = action.input if not lister else base.cut(action.input)
            desc = action.description if not lister else base.cut(action.description)
            data = (action.id, action.name, action.is_system, input_, desc, base.wrap(', '.join(tags)) or '<none>', action.created_at)
            if hasattr(action, 'updated_at'):
                data += (action.updated_at,)
            else:
                data += (None,)
        else:
            data = (tuple(('' for _ in range(len(ActionFormatter.COLUMNS)))),)
        return (ActionFormatter.headings(), data)