from typing import Dict, List
from libcloud.common.base import JsonResponse, ConnectionKey
class BuddyNSResponse(JsonResponse):
    errors = []
    objects = []

    def __init__(self, response, connection):
        super().__init__(response=response, connection=connection)
        self.errors, self.objects = self.parse_body_and_errors()
        if not self.success():
            raise BuddyNSException(code=self.status, message=self.errors.pop()['detail'])

    def parse_body_and_errors(self):
        js = super().parse_body()
        if 'detail' in js:
            self.errors.append(js)
        else:
            self.objects.append(js)
        return (self.errors, self.objects)

    def success(self):
        return len(self.errors) == 0