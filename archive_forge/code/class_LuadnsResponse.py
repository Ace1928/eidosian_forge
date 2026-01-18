import base64
from typing import Dict, List
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
class LuadnsResponse(JsonResponse):
    errors = []
    objects = []

    def __init__(self, response, connection):
        super().__init__(response=response, connection=connection)
        self.errors, self.objects = self.parse_body_and_errors()
        if not self.success():
            raise LuadnsException(code=self.status, message=self.errors.pop()['message'])

    def parse_body_and_errors(self):
        js = super().parse_body()
        if 'message' in js:
            self.errors.append(js)
        else:
            self.objects.append(js)
        return (self.errors, self.objects)

    def success(self):
        return len(self.errors) == 0