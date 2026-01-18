from six.moves import range
import sys
from pyu2f import apdu
from pyu2f import errors
class ApduTest(unittest.TestCase):

    def testSerializeCommandApdu(self):
        cmd = apdu.CommandApdu(0, 1, 3, 4, bytearray([16, 32, 48]))
        self.assertEqual(cmd.ToByteArray(), bytearray([0, 1, 3, 4, 0, 0, 3, 16, 32, 48, 0, 0]))
        self.assertEqual(cmd.ToLegacyU2FByteArray(), bytearray([0, 1, 3, 4, 0, 0, 3, 16, 32, 48, 0, 0]))

    def testSerializeCommandApduNoData(self):
        cmd = apdu.CommandApdu(0, 1, 3, 4)
        self.assertEqual(cmd.ToByteArray(), bytearray([0, 1, 3, 4, 0, 0, 0]))
        self.assertEqual(cmd.ToLegacyU2FByteArray(), bytearray([0, 1, 3, 4, 0, 0, 0, 0, 0]))

    def testSerializeCommandApduTooLong(self):
        self.assertRaises(errors.InvalidCommandError, apdu.CommandApdu, 0, 1, 3, 4, bytearray((0 for x in range(0, 65536))))

    def testResponseApduParse(self):
        resp = apdu.ResponseApdu(bytearray([5, 4, 144, 0]))
        self.assertEqual(resp.body, bytearray([5, 4]))
        self.assertEqual(resp.sw1, 144)
        self.assertEqual(resp.sw2, 0)
        self.assertTrue(resp.IsSuccess())

    def testResponseApduParseNoBody(self):
        resp = apdu.ResponseApdu(bytearray([105, 133]))
        self.assertEqual(resp.sw1, 105)
        self.assertEqual(resp.sw2, 133)
        self.assertFalse(resp.IsSuccess())

    def testResponseApduParseInvalid(self):
        self.assertRaises(errors.InvalidResponseError, apdu.ResponseApdu, bytearray([5]))

    def testResponseApduCheckSuccessTUPRequired(self):
        resp = apdu.ResponseApdu(bytearray([105, 133]))
        self.assertRaises(errors.TUPRequiredError, resp.CheckSuccessOrRaise)

    def testResponseApduCheckSuccessInvalidKeyHandle(self):
        resp = apdu.ResponseApdu(bytearray([106, 128]))
        self.assertRaises(errors.InvalidKeyHandleError, resp.CheckSuccessOrRaise)

    def testResponseApduCheckSuccessOtherError(self):
        resp = apdu.ResponseApdu(bytearray([250, 5]))
        self.assertRaises(errors.ApduError, resp.CheckSuccessOrRaise)