import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
class TestImportr(object):

    def test_importr_stats(self):
        stats = robjects.packages.importr('stats', on_conflict='warn')
        assert isinstance(stats, robjects.packages.Package)

    def test_import_stats_with_libloc(self):
        path = os.path.dirname(robjects.packages_utils.get_packagepath('stats'))
        stats = robjects.packages.importr('stats', on_conflict='warn', lib_loc=path)
        assert isinstance(stats, robjects.packages.Package)

    def test_import_stats_with_libloc_and_suppressmessages(self):
        path = os.path.dirname(robjects.packages_utils.get_packagepath('stats'))
        stats = robjects.packages.importr('stats', lib_loc=path, on_conflict='warn', suppress_messages=False)
        assert isinstance(stats, robjects.packages.Package)

    def test_import_stats_with_libloc_with_quote(self):
        path = 'coin"coin'
        with pytest.raises(robjects.packages.PackageNotInstalledError), pytest.warns(UserWarning):
            Tmp_File = io.StringIO
            tmp_file = Tmp_File()
            try:
                stdout = sys.stdout
                sys.stdout = tmp_file
                robjects.packages.importr('dummy_inexistant', lib_loc=path)
            finally:
                sys.stdout = stdout
                tmp_file.close()

    def test_import_datasets(self):
        datasets = robjects.packages.importr('datasets')
        assert isinstance(datasets, robjects.packages.Package)
        assert isinstance(datasets.__rdata__, robjects.packages.PackageData)
        assert isinstance(robjects.packages.data(datasets), robjects.packages.PackageData)

    def test_datatsets_names(self):
        datasets = robjects.packages.importr('datasets')
        datasets_data = robjects.packages.data(datasets)
        datasets_names = tuple(datasets_data.names())
        assert len(datasets_names) > 0
        assert all((isinstance(x, str) for x in datasets_names))

    def test_datatsets_fetch(self):
        datasets = robjects.packages.importr('datasets')
        datasets_data = robjects.packages.data(datasets)
        datasets_names = tuple(datasets_data.names())
        assert isinstance(datasets_data.fetch(datasets_names[0]), robjects.Environment)
        with pytest.raises(KeyError):
            datasets_data.fetch('foo_%s' % datasets_names[0])