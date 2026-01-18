from unittest import SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
from holoviews.element.comparison import ComparisonTestCase
class TestPandasTransforms(ComparisonTestCase):

    def setUp(self):
        import hvplot.pandas

    def test_pandas_transform(self):
        demo_df = pd.DataFrame({'value': np.random.randn(50), 'probability': np.random.rand(50)})
        percent = hv.dim('probability') * 100
        scatter = demo_df.hvplot.scatter(x='value', y='probability', transforms=dict(probability=percent))
        self.assertEqual(scatter.data['probability'].values, demo_df['probability'].values * 100)