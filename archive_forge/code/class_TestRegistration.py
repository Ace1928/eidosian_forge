from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
class TestRegistration:

    @pytest.mark.single_cpu
    def test_dont_register_by_default(self):
        code = 'import matplotlib.units; import pandas as pd; units = dict(matplotlib.units.registry); assert pd.Timestamp not in units'
        call = [sys.executable, '-c', code]
        assert subprocess.check_call(call) == 0

    def test_registering_no_warning(self):
        plt = pytest.importorskip('matplotlib.pyplot')
        s = Series(range(12), index=date_range('2017', periods=12))
        _, ax = plt.subplots()
        register_matplotlib_converters()
        ax.plot(s.index, s.values)
        plt.close()

    def test_pandas_plots_register(self):
        plt = pytest.importorskip('matplotlib.pyplot')
        s = Series(range(12), index=date_range('2017', periods=12))
        with tm.assert_produces_warning(None) as w:
            s.plot()
        try:
            assert len(w) == 0
        finally:
            plt.close()

    def test_matplotlib_formatters(self):
        units = pytest.importorskip('matplotlib.units')
        with cf.option_context('plotting.matplotlib.register_converters', True):
            with cf.option_context('plotting.matplotlib.register_converters', False):
                assert Timestamp not in units.registry
            assert Timestamp in units.registry

    def test_option_no_warning(self):
        pytest.importorskip('matplotlib.pyplot')
        ctx = cf.option_context('plotting.matplotlib.register_converters', False)
        plt = pytest.importorskip('matplotlib.pyplot')
        s = Series(range(12), index=date_range('2017', periods=12))
        _, ax = plt.subplots()
        with ctx:
            ax.plot(s.index, s.values)
        register_matplotlib_converters()
        with ctx:
            ax.plot(s.index, s.values)
        plt.close()

    def test_registry_resets(self):
        units = pytest.importorskip('matplotlib.units')
        dates = pytest.importorskip('matplotlib.dates')
        original = dict(units.registry)
        try:
            units.registry.clear()
            date_converter = dates.DateConverter()
            units.registry[datetime] = date_converter
            units.registry[date] = date_converter
            register_matplotlib_converters()
            assert units.registry[date] is not date_converter
            deregister_matplotlib_converters()
            assert units.registry[date] is date_converter
        finally:
            units.registry.clear()
            for k, v in original.items():
                units.registry[k] = v