import argparse
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class WorkbookFormatter(base.MistralFormatter):
    COLUMNS = [('name', 'Name'), ('namespace', 'Namespace'), ('tags', 'Tags'), ('scope', 'Scope'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]

    @staticmethod
    def format(workbook=None, lister=False):
        if workbook:
            data = (workbook.name, workbook.namespace, base.wrap(', '.join(workbook.tags or '')) or '<none>', workbook.scope, workbook.created_at)
            if hasattr(workbook, 'updated_at'):
                data += (workbook.updated_at,)
            else:
                data += (None,)
        else:
            data = (tuple(('' for _ in range(len(WorkbookFormatter.COLUMNS)))),)
        return (WorkbookFormatter.headings(), data)