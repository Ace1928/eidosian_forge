from boto.beanstalk.layer1 import Layer1
import boto.beanstalk.response
from boto.exception import BotoServerError
import boto.beanstalk.exception as exception
def _wrapped_low_level_api(*args, **kwargs):
    try:
        response = func(*args, **kwargs)
    except BotoServerError as e:
        raise exception.simple(e)
    cls_name = ''.join([part.capitalize() for part in name.split('_')]) + 'Response'
    cls = getattr(boto.beanstalk.response, cls_name)
    return cls(response)