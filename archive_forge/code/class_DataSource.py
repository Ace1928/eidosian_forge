from rpy2 import robjects
from rpy2.robjects.packages import (importr,
import warnings
class DataSource(robjects.ListVector):
    """ Source of data tables (e.g., in a schema in a relational database). """

    @property
    def tablenames(self):
        """ Call the R function dplyr::src_tbls() and return a vector
        of table names."""
        return tuple(dplyr.src_tbls(self))

    def get_table(self, name):
        """ "Get" table from a source (R dplyr's function `tbl`). """
        return DataFrame(tbl(self, name))