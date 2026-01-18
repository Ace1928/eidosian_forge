from unittest import TestCase
import simplejson as json
class RecursiveJSONEncoder(json.JSONEncoder):
    recurse = False

    def default(self, o):
        if o is JSONTestObject:
            if self.recurse:
                return [JSONTestObject]
            else:
                return 'JSONTestObject'
        return json.JSONEncoder.default(o)