from rdkit.ML.DecTree import Tree
def ClearExamples(self):
    self.examples = []
    self.badExamples = []
    self.trainingExamples = []
    self.testExamples = []
    for child in self.GetChildren():
        child.ClearExamples()