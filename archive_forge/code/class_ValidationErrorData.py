from collections import namedtuple
import json
class ValidationErrorData(namedtuple('ValidationErrorData', ['datum', 'schema', 'field'])):

    def __str__(self):
        if self.datum is None:
            return f'Field({self.field}) is None expected {self.schema}'
        return f'{self.field} is <{self.datum}> of type ' + f'{type(self.datum)} expected {self.schema}'