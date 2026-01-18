from pyparsing import ParserElement,LineEnd,Optional,Word,nums,Regex,\
class TAPSummary(object):

    def __init__(self, results):
        self.passedTests = []
        self.failedTests = []
        self.skippedTests = []
        self.todoTests = []
        self.bonusTests = []
        self.bail = False
        if results.plan:
            expected = list(range(1, int(results.plan.ubound) + 1))
        else:
            expected = list(range(1, len(results.tests) + 1))
        for i, res in enumerate(results.tests):
            if res.BAIL:
                self.bail = True
                self.skippedTests += [TAPTest.bailedTest(ii) for ii in expected[i:]]
                self.bailReason = res.reason
                break
            testnum = i + 1
            if res.testNumber != '':
                if testnum != int(res.testNumber):
                    print('ERROR! test %(testNumber)s out of sequence' % res)
                testnum = int(res.testNumber)
            res['testNumber'] = testnum
            test = TAPTest(res)
            if test.passed:
                self.passedTests.append(test)
            else:
                self.failedTests.append(test)
            if test.skipped:
                self.skippedTests.append(test)
            if test.todo:
                self.todoTests.append(test)
            if test.todo and test.passed:
                self.bonusTests.append(test)
        self.passedSuite = not self.bail and set(self.failedTests) - set(self.todoTests) == set()

    def summary(self, showPassed=False, showAll=False):
        testListStr = lambda tl: '[' + ','.join((str(t.num) for t in tl)) + ']'
        summaryText = []
        if showPassed or showAll:
            summaryText.append('PASSED: %s' % testListStr(self.passedTests))
        if self.failedTests or showAll:
            summaryText.append('FAILED: %s' % testListStr(self.failedTests))
        if self.skippedTests or showAll:
            summaryText.append('SKIPPED: %s' % testListStr(self.skippedTests))
        if self.todoTests or showAll:
            summaryText.append('TODO: %s' % testListStr(self.todoTests))
        if self.bonusTests or showAll:
            summaryText.append('BONUS: %s' % testListStr(self.bonusTests))
        if self.passedSuite:
            summaryText.append('PASSED')
        else:
            summaryText.append('FAILED')
        return '\n'.join(summaryText)