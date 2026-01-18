import re
import numpy as np
def _expect(self, token, tp):
    if not token.type == tp:
        raise SyntaxError()