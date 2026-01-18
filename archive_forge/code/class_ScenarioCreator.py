import pyomo.environ as pyo
class ScenarioCreator(object):
    """Create scenarios from parmest.

    Args:
        pest (Estimator): the parmest object
        solvername (str): name of the solver (e.g. "ipopt")

    """

    def __init__(self, pest, solvername):
        self.pest = pest
        self.solvername = solvername

    def ScenariosFromExperiments(self, addtoSet):
        """Creates new self.Scenarios list using the experiments only.

        Args:
            addtoSet (ScenarioSet): the scenarios will be added to this set
        Returns:
            a ScenarioSet
        """
        assert isinstance(addtoSet, ScenarioSet)
        scenario_numbers = list(range(len(self.pest.callback_data)))
        prob = 1.0 / len(scenario_numbers)
        for exp_num in scenario_numbers:
            model = self.pest._instance_creation_callback(exp_num, self.pest.callback_data)
            opt = pyo.SolverFactory(self.solvername)
            results = opt.solve(model)
            ThetaVals = dict()
            for theta in self.pest.theta_names:
                tvar = eval('model.' + theta)
                tval = pyo.value(tvar)
                ThetaVals[theta] = tval
            addtoSet.addone(ParmestScen('ExpScen' + str(exp_num), ThetaVals, prob))

    def ScenariosFromBootstrap(self, addtoSet, numtomake, seed=None):
        """Creates new self.Scenarios list using the experiments only.

        Args:
            addtoSet (ScenarioSet): the scenarios will be added to this set
            numtomake (int) : number of scenarios to create
        """
        assert isinstance(addtoSet, ScenarioSet)
        bootstrap_thetas = self.pest.theta_est_bootstrap(numtomake, seed=seed)
        addtoSet.append_bootstrap(bootstrap_thetas)