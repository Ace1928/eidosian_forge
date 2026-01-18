from pyomo.environ import (
from pyomo.common.collections import ComponentMap
class EightProcessFlowsheet(ConcreteModel):
    """Flowsheet for the 8 process problem."""

    def __init__(self, convex=True, *args, **kwargs):
        """Create the flowsheet."""
        kwargs.setdefault('name', 'DuranEx3')
        super(EightProcessFlowsheet, self).__init__(*args, **kwargs)
        m = self
        'Set declarations'
        I = m.I = RangeSet(2, 25, doc='process streams')
        J = m.J = RangeSet(1, 8, doc='process units')
        m.PI = RangeSet(1, 4, doc='integer constraints')
        m.DS = RangeSet(1, 4, doc='design specifications')
        '\n        1: Unit 8\n        2: Unit 8\n        3: Unit 4\n        4: Unit 4\n        '
        m.MB = RangeSet(1, 7, doc='mass balances')
        'Material balances:\n        1: 4-6-7\n        2: 3-5-8\n        3: 4-5\n        4: 1-2\n        5: 1-2-3\n        6: 6-7-4\n        7: 6-7\n        '
        'Parameter and initial point declarations'
        fixed_cost = {1: 5, 2: 8, 3: 6, 4: 10, 5: 6, 6: 7, 7: 4, 8: 5}
        CF = m.CF = Param(J, initialize=fixed_cost)
        variable_cost = {3: -10, 5: -15, 9: -40, 19: 25, 21: 35, 25: -35, 17: 80, 14: 15, 10: 15, 2: 1, 4: 1, 18: -65, 20: -60, 22: -80}
        CV = m.CV = Param(I, initialize=variable_cost, default=0)
        initY = {'sub1': {1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 1}, 'sub2': {1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0, 8: 1}, 'sub3': {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 1}}
        initX = {2: 2, 3: 1.5, 4: 0, 5: 0, 6: 0.75, 7: 0.5, 8: 0.5, 9: 0.75, 10: 0, 11: 1.5, 12: 1.34, 13: 2, 14: 2.5, 15: 0, 16: 0, 17: 2, 18: 0.75, 19: 2, 20: 1.5, 21: 0, 22: 0, 23: 1.7, 24: 1.5, 25: 0.5}
        'Variable declarations'
        Y = m.Y = Var(J, domain=Binary, initialize=initY['sub1'])
        X = m.X = Var(I, domain=NonNegativeReals, initialize=initX)
        CONSTANT = m.constant = Param(initialize=122.0)
        'Constraint definitions'
        m.inout3 = Constraint(expr=1.5 * m.X[9] + m.X[10] == m.X[8])
        m.inout4 = Constraint(expr=1.25 * (m.X[12] + m.X[14]) == m.X[13])
        m.inout5 = Constraint(expr=m.X[15] == 2 * m.X[16])
        if convex:
            m.inout1 = Constraint(expr=exp(m.X[3]) - 1 <= m.X[2])
            m.inout2 = Constraint(expr=exp(m.X[5] / 1.2) - 1 <= m.X[4])
            m.inout7 = Constraint(expr=exp(m.X[22]) - 1 <= m.X[21])
            m.inout8 = Constraint(expr=exp(m.X[18]) - 1 <= m.X[10] + m.X[17])
            m.inout6 = Constraint(expr=exp(m.X[20] / 1.5) - 1 <= m.X[19])
        else:
            m.inout1 = Constraint(expr=exp(m.X[3]) - 1 == m.X[2])
            m.inout2 = Constraint(expr=exp(m.X[5] / 1.2) - 1 == m.X[4])
            m.inout7 = Constraint(expr=exp(m.X[22]) - 1 == m.X[21])
            m.inout8 = Constraint(expr=exp(m.X[18]) - 1 == m.X[10] + m.X[17])
            m.inout6 = Constraint(expr=exp(m.X[20] / 1.5) - 1 == m.X[19])
        m.massbal1 = Constraint(expr=m.X[13] == m.X[19] + m.X[21])
        m.massbal2 = Constraint(expr=m.X[17] == m.X[9] + m.X[16] + m.X[25])
        m.massbal3 = Constraint(expr=m.X[11] == m.X[12] + m.X[15])
        m.massbal4 = Constraint(expr=m.X[3] + m.X[5] == m.X[6] + m.X[11])
        m.massbal5 = Constraint(expr=m.X[6] == m.X[7] + m.X[8])
        m.massbal6 = Constraint(expr=m.X[23] == m.X[20] + m.X[22])
        m.massbal7 = Constraint(expr=m.X[23] == m.X[14] + m.X[24])
        m.specs1 = Constraint(expr=m.X[10] <= 0.8 * m.X[17])
        m.specs2 = Constraint(expr=m.X[10] >= 0.4 * m.X[17])
        m.specs3 = Constraint(expr=m.X[12] <= 5 * m.X[14])
        m.specs4 = Constraint(expr=m.X[12] >= 2 * m.X[14])
        m.logical1 = Constraint(expr=m.X[2] <= 10 * m.Y[1])
        m.logical2 = Constraint(expr=m.X[4] <= 10 * m.Y[2])
        m.logical3 = Constraint(expr=m.X[9] <= 10 * m.Y[3])
        m.logical4 = Constraint(expr=m.X[12] + m.X[14] <= 10 * m.Y[4])
        m.logical5 = Constraint(expr=m.X[15] <= 10 * m.Y[5])
        m.logical6 = Constraint(expr=m.X[19] <= 10 * m.Y[6])
        m.logical7 = Constraint(expr=m.X[21] <= 10 * m.Y[7])
        m.logical8 = Constraint(expr=m.X[10] + m.X[17] <= 10 * m.Y[8])
        m.pureint1 = Constraint(expr=m.Y[1] + m.Y[2] == 1)
        m.pureint2 = Constraint(expr=m.Y[4] + m.Y[5] <= 1)
        m.pureint3 = Constraint(expr=m.Y[6] + m.Y[7] - m.Y[4] == 0)
        m.pureint4 = Constraint(expr=m.Y[3] - m.Y[8] <= 0)
        'Cost (objective) function definition'
        m.objective = Objective(expr=sum((Y[j] * CF[j] for j in J)) + sum((X[i] * CV[i] for i in I)) + CONSTANT, sense=minimize)
        'Bound definitions'
        x_ubs = {2: 10, 3: 2, 4: 10, 5: 2, 9: 2, 10: 1, 14: 1, 17: 2, 18: 10, 19: 2, 20: 10, 21: 2, 22: 10, 25: 3}
        for i, x_ub in x_ubs.items():
            X[i].setub(x_ub)
        m.optimal_value = 68.0097
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.X[2]] = 0.0
        m.optimal_solution[m.X[3]] = 0.0
        m.optimal_solution[m.X[4]] = 4.294490050470028
        m.optimal_solution[m.X[5]] = 1.9999999999999998
        m.optimal_solution[m.X[6]] = 0.6666666666666665
        m.optimal_solution[m.X[7]] = 0.1979983174202069
        m.optimal_solution[m.X[8]] = 0.4686683492464596
        m.optimal_solution[m.X[9]] = 0.0
        m.optimal_solution[m.X[10]] = 0.4686683492464596
        m.optimal_solution[m.X[11]] = 1.3333333333333333
        m.optimal_solution[m.X[12]] = 1.3333333333333333
        m.optimal_solution[m.X[13]] = 2.0
        m.optimal_solution[m.X[14]] = 0.26666666666666666
        m.optimal_solution[m.X[15]] = 0.0
        m.optimal_solution[m.X[16]] = 0.0
        m.optimal_solution[m.X[17]] = 0.5858354365580745
        m.optimal_solution[m.X[18]] = 0.720035498875516
        m.optimal_solution[m.X[19]] = 2.0
        m.optimal_solution[m.X[20]] = 1.6479184330021648
        m.optimal_solution[m.X[21]] = 0.0
        m.optimal_solution[m.X[22]] = 0.0
        m.optimal_solution[m.X[23]] = 1.6479184330021648
        m.optimal_solution[m.X[24]] = 1.3812517663354982
        m.optimal_solution[m.X[25]] = 0.5858354365580745
        m.optimal_solution[m.Y[1]] = 0.0
        m.optimal_solution[m.Y[2]] = 1.0
        m.optimal_solution[m.Y[3]] = 0.0
        m.optimal_solution[m.Y[4]] = 1.0
        m.optimal_solution[m.Y[5]] = 0.0
        m.optimal_solution[m.Y[6]] = 1.0
        m.optimal_solution[m.Y[7]] = 0.0
        m.optimal_solution[m.Y[8]] = 1.0