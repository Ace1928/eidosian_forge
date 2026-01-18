import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag

        Check if we should add or remove any rules from consideration,
        given the changes made by *rule*.
        