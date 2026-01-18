from boto.pyami.config import Config
from boto.services.message import ServiceMessage
import boto
class ServiceDef(Config):

    def __init__(self, config_file, aws_access_key_id=None, aws_secret_access_key=None):
        super(ServiceDef, self).__init__(config_file)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        script = Config.get(self, 'Pyami', 'scripts')
        if script:
            self.name = script.split('.')[-1]
        else:
            self.name = None

    def get(self, name, default=None):
        return super(ServiceDef, self).get(self.name, name, default)

    def has_option(self, option):
        return super(ServiceDef, self).has_option(self.name, option)

    def getint(self, option, default=0):
        try:
            val = super(ServiceDef, self).get(self.name, option)
            val = int(val)
        except:
            val = int(default)
        return val

    def getbool(self, option, default=False):
        try:
            val = super(ServiceDef, self).get(self.name, option)
            if val.lower() == 'true':
                val = True
            else:
                val = False
        except:
            val = default
        return val

    def get_obj(self, name):
        """
        Returns the AWS object associated with a given option.

        The heuristics used are a bit lame.  If the option name contains
        the word 'bucket' it is assumed to be an S3 bucket, if the name
        contains the word 'queue' it is assumed to be an SQS queue and
        if it contains the word 'domain' it is assumed to be a SimpleDB
        domain.  If the option name specified does not exist in the
        config file or if the AWS object cannot be retrieved this
        returns None.
        """
        val = self.get(name)
        if not val:
            return None
        if name.find('queue') >= 0:
            obj = boto.lookup('sqs', val)
            if obj:
                obj.set_message_class(ServiceMessage)
        elif name.find('bucket') >= 0:
            obj = boto.lookup('s3', val)
        elif name.find('domain') >= 0:
            obj = boto.lookup('sdb', val)
        else:
            obj = None
        return obj