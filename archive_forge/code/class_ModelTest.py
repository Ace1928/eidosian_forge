import json
import sys
from pyu2f import errors
from pyu2f import model
class ModelTest(unittest.TestCase):

    def testClientDataRegistration(self):
        cd = model.ClientData(model.ClientData.TYP_REGISTRATION, b'ABCD', 'somemachine')
        obj = json.loads(cd.GetJson())
        self.assertEquals(len(list(obj.keys())), 3)
        self.assertEquals(obj['typ'], model.ClientData.TYP_REGISTRATION)
        self.assertEquals(obj['challenge'], 'QUJDRA')
        self.assertEquals(obj['origin'], 'somemachine')

    def testClientDataAuth(self):
        cd = model.ClientData(model.ClientData.TYP_AUTHENTICATION, b'ABCD', 'somemachine')
        obj = json.loads(cd.GetJson())
        self.assertEquals(len(list(obj.keys())), 3)
        self.assertEquals(obj['typ'], model.ClientData.TYP_AUTHENTICATION)
        self.assertEquals(obj['challenge'], 'QUJDRA')
        self.assertEquals(obj['origin'], 'somemachine')

    def testClientDataInvalid(self):
        self.assertRaises(errors.InvalidModelError, model.ClientData, 'foobar', b'ABCD', 'somemachine')