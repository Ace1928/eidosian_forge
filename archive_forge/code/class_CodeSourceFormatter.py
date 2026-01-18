import argparse
from cliff import show
from osc_lib.command import command
from mistralclient.commands.v2 import base
from mistralclient import utils
class CodeSourceFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('name', 'Name'), ('namespace', 'Namespace'), ('project_id', 'Project ID'), ('scope', 'Scope'), ('created_at', 'Created at'), ('updated_at', 'Updated at')]

    @staticmethod
    def format(code_src=None, lister=False):
        if code_src:
            data = (code_src.id, code_src.name, code_src.namespace, code_src.project_id, code_src.scope, code_src.created_at)
            if hasattr(code_src, 'updated_at'):
                data += (code_src.updated_at,)
            else:
                data += (None,)
        else:
            data = (('',) * len(CodeSourceFormatter.COLUMNS),)
        return (CodeSourceFormatter.headings(), data)