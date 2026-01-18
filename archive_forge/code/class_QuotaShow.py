from cliff import show
from aodhclient import exceptions
from aodhclient import utils
class QuotaShow(show.ShowOne):
    """Show quota for a project"""

    def get_parser(self, prog_name):
        parser = super(QuotaShow, self).get_parser(prog_name)
        parser.add_argument('--project', help='Project ID. If not specified, get quota for the current project.')
        return parser

    def take_action(self, parsed_args):
        c = utils.get_client(self)
        quota = c.quota.list(project=parsed_args.project)
        ret = {}
        for q in quota['quotas']:
            ret[q['resource']] = q['limit']
        return self.dict2columns(ret)