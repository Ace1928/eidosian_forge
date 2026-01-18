from io import StringIO
import os
from pathlib import Path
import pytest
from pandas.errors import ParserError
import pandas._testing as tm
from pandas.io.parsers import read_csv
import pandas.io.parsers.readers as parsers
class NoNextBuffer:

    def __init__(self, csv_data) -> None:
        self.data = csv_data

    def __next__(self):
        return self.data.__next__()

    def read(self):
        return self.data

    def readline(self):
        return self.data