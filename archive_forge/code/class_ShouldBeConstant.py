import textwrap
class ShouldBeConstant(VarLibMergeError):
    """some values were different, but should have been the same"""

    @property
    def details(self):
        basic_message = super().details
        if self.stack[0] != '.FeatureCount' or self.merger is None:
            return basic_message
        assert self.stack[0] == '.FeatureCount'
        offender_index, _ = self.offender
        bad_ttf = self.merger.ttfs[offender_index]
        good_ttf = next((ttf for ttf in self.merger.ttfs if self.stack[-1] in ttf and ttf[self.stack[-1]].table.FeatureList.FeatureCount == self.cause['expected']))
        good_features = [x.FeatureTag for x in good_ttf[self.stack[-1]].table.FeatureList.FeatureRecord]
        bad_features = [x.FeatureTag for x in bad_ttf[self.stack[-1]].table.FeatureList.FeatureRecord]
        return basic_message + f'\nIncompatible features between masters.\nExpected: {', '.join(good_features)}.\nGot: {', '.join(bad_features)}.\n'