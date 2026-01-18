import os
import pickle
import sys
import numpy
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import InfoTheory
def SupplierFromDetails(details):
    from rdkit.VLib.NodeLib.DbMolSupply import DbMolSupplyNode
    from rdkit.VLib.NodeLib.SmilesSupply import SmilesSupplyNode
    if details.dbName:
        conn = DbConnect(details.dbName, details.tableName)
        suppl = DbMolSupplyNode(conn.GetData())
    else:
        suppl = SmilesSupplyNode(details.inFileName, delim=details.delim, nameColumn=details.nameCol, smilesColumn=details.smiCol, titleLine=details.hasTitle)
        if isinstance(details.actCol, int):
            suppl.reset()
            m = next(suppl)
            actName = m.GetPropNames()[details.actCol]
            details.actCol = actName
        if isinstance(details.nameCol, int):
            suppl.reset()
            m = next(suppl)
            nameName = m.GetPropNames()[details.nameCol]
            details.nameCol = nameName
            suppl.reset()
    if isinstance(details.actCol, int):
        suppl.reset()
        m = next(suppl)
        actName = m.GetPropNames()[details.actCol]
        details.actCol = actName
    if isinstance(details.nameCol, int):
        suppl.reset()
        m = next(suppl)
        nameName = m.GetPropNames()[details.nameCol]
        details.nameCol = nameName
        suppl.reset()
    return suppl