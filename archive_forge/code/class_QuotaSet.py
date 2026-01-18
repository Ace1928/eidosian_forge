from cliff import show
from aodhclient import exceptions
from aodhclient import utils
class QuotaSet(show.ShowOne):

    def get_parser(self, prog_name):
        parser = super(QuotaSet, self).get_parser(prog_name)
        parser.add_argument('project', help='Project ID.')
        parser.add_argument('--alarm', type=int, help='New value for the alarm quota. Value -1 means unlimited.')
        return parser

    def take_action(self, parsed_args):
        resource_quotas = []
        if parsed_args.alarm is not None:
            if parsed_args.alarm < -1:
                raise exceptions.CommandError('Quota limit cannot be less than -1.')
            resource_quotas.append({'resource': 'alarms', 'limit': parsed_args.alarm})
        c = utils.get_client(self)
        quota = c.quota.create(parsed_args.project, resource_quotas)
        ret = {}
        for q in quota['quotas']:
            ret[q['resource']] = q['limit']
        return self.dict2columns(ret)