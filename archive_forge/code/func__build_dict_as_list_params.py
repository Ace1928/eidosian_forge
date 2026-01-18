import uuid
import hashlib
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.compat import json
import boto
def _build_dict_as_list_params(self, params, dictionary, name):
    """
            Serialize a parameter 'name' which value is a 'dictionary' into a list of parameters.

            See: http://docs.aws.amazon.com/sns/latest/api/API_SetPlatformApplicationAttributes.html
            For example::

                dictionary = {'PlatformPrincipal': 'foo', 'PlatformCredential': 'bar'}
                name = 'Attributes'

            would result in params dict being populated with:
                Attributes.entry.1.key    = PlatformPrincipal
                Attributes.entry.1.value  = foo
                Attributes.entry.2.key    = PlatformCredential
                Attributes.entry.2.value  = bar

      :param params: the resulting parameters will be added to this dict
      :param dictionary: dict - value of the serialized parameter
      :param name: name of the serialized parameter
      """
    items = sorted(dictionary.items(), key=lambda x: x[0])
    for kv, index in zip(items, list(range(1, len(items) + 1))):
        key, value = kv
        prefix = '%s.entry.%s' % (name, index)
        params['%s.key' % prefix] = key
        params['%s.value' % prefix] = value