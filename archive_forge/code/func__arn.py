from boto.compat import urllib
from boto.sqs.message import Message
def _arn(self):
    parts = self.id.split('/')
    if self.connection.region.name == 'cn-north-1':
        partition = 'aws-cn'
    else:
        partition = 'aws'
    return 'arn:%s:sqs:%s:%s:%s' % (partition, self.connection.region.name, parts[1], parts[2])