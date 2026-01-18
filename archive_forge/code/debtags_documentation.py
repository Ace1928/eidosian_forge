import pickle
import re
from debian.deprecation import function_deprecated_by

        Generate the list of correlation as a tuple (hastag, hasalsotag, score).

        Every tuple will indicate that the tag 'hastag' tends to also
        have 'hasalsotag' with a score of 'score'.
        