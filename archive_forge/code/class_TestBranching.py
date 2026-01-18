import yaql.tests
class TestBranching(yaql.tests.TestCase):

    def test_switch(self):
        expr = 'switch($ < 10 => 1, $ >= 10 and $ < 100 => 2, $ >= 100 => 3)'
        self.assertEqual(3, self.eval(expr, data=123))
        self.assertEqual(2, self.eval(expr, data=50))
        self.assertEqual(1, self.eval(expr, data=-123))

    def test_select_case(self):
        expr = 'selectCase($ < 10, $ >= 10 and $ < 100)'
        self.assertEqual(2, self.eval(expr, data=123))
        self.assertEqual(1, self.eval(expr, data=50))
        self.assertEqual(0, self.eval(expr, data=-123))

    def test_select_all_cases(self):
        expr = 'selectAllCases($ < 10, $ > 5)'
        self.assertEqual([0], self.eval(expr, data=1))
        self.assertEqual([0, 1], self.eval(expr, data=7))
        self.assertEqual([1], self.eval(expr, data=12))

    def test_examine(self):
        expr = 'examine($ < 10, $ > 5)'
        self.assertEqual([True, False], self.eval(expr, data=1))
        self.assertEqual([True, True], self.eval(expr, data=7))
        self.assertEqual([False, True], self.eval(expr, data=12))

    def test_switch_case(self):
        expr = "$.switchCase('a', 'b', 'c')"
        self.assertEqual('a', self.eval(expr, data=0))
        self.assertEqual('b', self.eval(expr, data=1))
        self.assertEqual('c', self.eval(expr, data=3))
        self.assertEqual('c', self.eval(expr, data=30))
        self.assertEqual('c', self.eval(expr, data=-30))

    def test_coalesce(self):
        self.assertEqual(2, self.eval('coalesce($, 2)', data=None))
        self.assertEqual(1, self.eval('coalesce($, 2)', data=1))
        self.assertEqual(2, self.eval('coalesce($, $, 2)', data=None))
        self.assertIsNone(self.eval('coalesce($)', data=None))