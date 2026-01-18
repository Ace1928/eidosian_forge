import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
Convert the data in a ``nodelist`` into a networkx labeled directed graph.