def AddSupplementalData(self, data):
    if not hasattr(self, '_supplementalData'):
        self.SetSupplementalData([])
    self._supplementalData.append(data)