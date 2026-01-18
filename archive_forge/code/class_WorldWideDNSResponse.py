import re
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class WorldWideDNSResponse(Response):

    def parse_body(self):
        """
        Parse response body.

        :return: Parsed body.
        :rtype: ``str``
        """
        if self._code_response(self.body):
            codes = re.split('\r?\n', self.body)
            for code in codes:
                if code in OK_CODES:
                    continue
                elif code in ERROR_CODES:
                    exception = ERROR_CODE_TO_EXCEPTION_CLS.get(code)
                    if code in ['411', '412', '413']:
                        server = int(code[2])
                        raise exception(server, self.status)
                    raise exception(self.status)
        return self.body

    def _code_response(self, body):
        """
        Checks if the response body contains code status.

        :rtype: ``bool``
        """
        available_response_codes = OK_CODES + ERROR_CODES
        codes = re.split('\r?\n', body)
        if codes[0] in available_response_codes:
            return True
        return False