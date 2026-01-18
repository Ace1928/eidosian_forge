import os
import json
import os.path
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.scripting.pyomo_main import main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL
import pyomo.neos
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
class PyomoCommandDriver(object):

    def _run(self, opt, constrained=True):
        expected_y = {(pyo.minimize, True): -1, (pyo.maximize, True): 1, (pyo.minimize, False): -10, (pyo.maximize, False): 10}[self.sense, constrained]
        filename = 'model_min_lp.py' if self.sense == pyo.minimize else 'model_max_lp.py'
        results = os.path.join(currdir, 'result.json')
        args = ['solve', os.path.join(currdir, filename), '--solver-manager=neos', '--solver=%s' % opt, '--logging=quiet', '--save-results=%s' % results, '--results-format=json', '-c']
        try:
            output = main(args)
            self.assertEqual(output.errorcode, 0)
            with open(results) as FILE:
                data = json.load(FILE)
        finally:
            cleanup()
            if os.path.exists(results):
                os.remove(results)
        self.assertEqual(data['Solver'][0]['Status'], 'ok')
        self.assertEqual(data['Solution'][1]['Status'], 'optimal')
        self.assertAlmostEqual(data['Solution'][1]['Objective']['obj']['Value'], expected_y, delta=1e-05)
        if constrained:
            self.assertAlmostEqual(data['Solution'][1]['Variable']['x']['Value'], 1, delta=1e-05)
        self.assertAlmostEqual(data['Solution'][1]['Variable']['y']['Value'], expected_y, delta=1e-05)