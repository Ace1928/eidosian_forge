from typing import NamedTuple
from google.api_core.exceptions import InvalidArgument
class CloudRegion(NamedTuple):
    name: str

    def __str__(self):
        return self.name

    @staticmethod
    def parse(to_parse: str):
        splits = to_parse.split('-')
        if len(splits) != 2:
            raise InvalidArgument('Invalid region name: ' + to_parse)
        return CloudRegion(name=splits[0] + '-' + splits[1])

    @property
    def region(self):
        return self