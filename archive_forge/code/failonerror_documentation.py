import pytest
from petl.test.helpers import ieq, eq_
import petl.config as config
In the input rows, the first row should process through the
    transformation cleanly.  The second row should generate an
    exception.  There are no requirements for any other rows.