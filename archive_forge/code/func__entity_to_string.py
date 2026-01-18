import pprint
from abc import abstractmethod
def _entity_to_string(self, entity):
    return ', '.join([f'{key}={self.to_string(value)}' for key, value in entity])