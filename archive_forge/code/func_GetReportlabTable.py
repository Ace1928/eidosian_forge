def GetReportlabTable(self, *args, **kwargs):
    """ this becomes a method of DbConnect  """
    dbRes = self.GetData(*args, **kwargs)
    rawD = [dbRes.GetColumnNames()]
    colTypes = dbRes.GetColumnTypes()
    binCols = []
    for i in range(len(colTypes)):
        if colTypes[i] in DbInfo.sqlBinTypes or colTypes[i] == 'binary':
            binCols.append(i)
    nRows = 0
    for entry in dbRes:
        nRows += 1
        for col in binCols:
            entry = list(entry)
            entry[col] = 'N/A'
        rawD.append(entry)
    res = platypus.Table(rawD)
    return res