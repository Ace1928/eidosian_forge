from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
@unittest.skipIf(not parmest.parmest_available, 'Cannot test parmest: required dependencies are missing')
@unittest.skipIf(not graphics.imports_available, 'parmest.graphics imports are unavailable')
@unittest.skipIf(is_osx, 'Disabling graphics tests on OSX due to issue in Matplotlib, see Pyomo PR #1337')
class TestGraphics(unittest.TestCase):

    def setUp(self):
        self.A = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
        self.B = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))

    def test_pairwise_plot(self):
        graphics.pairwise_plot(self.A, alpha=0.8, distributions=['Rect', 'MVN', 'KDE'])

    def test_grouped_boxplot(self):
        graphics.grouped_boxplot(self.A, self.B, normalize=True, group_names=['A', 'B'])

    def test_grouped_violinplot(self):
        graphics.grouped_violinplot(self.A, self.B)