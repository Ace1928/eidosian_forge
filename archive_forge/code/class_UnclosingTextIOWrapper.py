import re
import sys
class UnclosingTextIOWrapper(TextIOWrapper):

    def close(self):
        self.flush()