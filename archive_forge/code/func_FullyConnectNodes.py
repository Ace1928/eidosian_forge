import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def FullyConnectNodes(self):
    """ Fully connects each layer in the network to the one above it


     **Note**
       this sets the connections, but does not assign weights

    """
    nodeList = list(range(self.numInputNodes))
    nConnections = 0
    for layer in range(self.numHiddenLayers):
        for i in self.layerIndices[layer + 1]:
            self.nodeList[i].SetInputs(nodeList)
            nConnections = nConnections + len(nodeList)
        nodeList = self.layerIndices[layer + 1]
    for i in self.layerIndices[-1]:
        self.nodeList[i].SetInputs(nodeList)
        nConnections = nConnections + len(nodeList)
    self.nConnections = nConnections