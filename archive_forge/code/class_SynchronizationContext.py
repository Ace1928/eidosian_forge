import base64
import hashlib
import io
import struct
import time
import zipfile
from vtk.vtkCommonCore import vtkTypeInt32Array, vtkTypeUInt32Array
from vtk.vtkCommonDataModel import vtkDataObject
from vtk.vtkFiltersGeometry import (
from vtk.vtkRenderingCore import vtkColorTransferFunction
from .enums import TextPosition
class SynchronizationContext:

    def __init__(self, id_root=None, serialize_all_data_arrays=False, debug=False):
        self.serializeAllDataArrays = serialize_all_data_arrays
        self.dataArrayCache = {}
        self.lastDependenciesMapping = {}
        self.ingoreLastDependencies = False
        self.idRoot = id_root
        self.debugSerializers = debug
        self.debugAll = debug
        self.annotations = {}

    def getReferenceId(self, instance):
        if not self.idRoot or (hasattr(instance, 'IsA') and instance.IsA('vtkCamera')):
            return getReferenceId(instance)
        else:
            return self.idRoot + getReferenceId(instance)

    def addAnnotation(self, parent, prop, propId):
        if prop.GetClassName() == 'vtkCornerAnnotation':
            annotation = {'id': propId, 'viewport': parent.GetViewport(), 'fontSize': prop.GetLinearFontScaleFactor() * 2, 'fontFamily': prop.GetTextProperty().GetFontFamilyAsString(), 'color': prop.GetTextProperty().GetColor(), **{pos.name: prop.GetText(pos.value) for pos in TextPosition}}
            if self.annotations is None:
                self.annotations = {propId: annotation}
            else:
                self.annotations.update({propId: annotation})

    def getAnnotations(self):
        return list(self.annotations.values())

    def setIgnoreLastDependencies(self, force):
        self.ingoreLastDependencies = force

    def cacheDataArray(self, pMd5, data):
        self.dataArrayCache[pMd5] = data

    def getCachedDataArray(self, pMd5, binary=False, compression=False):
        cacheObj = self.dataArrayCache[pMd5]
        array = cacheObj['array']
        cacheTime = cacheObj['mTime']
        if cacheTime != array.GetMTime():
            if context.debugAll:
                print(' ***** ERROR: you asked for an old cache key! ***** ')
        if array.GetDataType() in (12, 16, 17):
            arraySize = array.GetNumberOfTuples() * array.GetNumberOfComponents()
            if array.GetDataType() in (12, 17):
                newArray = vtkTypeUInt32Array()
            else:
                newArray = vtkTypeInt32Array()
            newArray.SetNumberOfTuples(arraySize)
            for i in range(arraySize):
                newArray.SetValue(i, -1 if array.GetValue(i) < 0 else array.GetValue(i))
            pBuffer = buffer(newArray)
        else:
            pBuffer = buffer(array)
        if binary:
            return pBuffer.tobytes() if not compression else zipCompression(pMd5, pBuffer.tobytes())
        return base64Encode(pBuffer if not compression else zipCompression(pMd5, pBuffer.tobytes()))

    def checkForArraysToRelease(self, timeWindow=20):
        cutOffTime = time.time() - timeWindow
        shasToDelete = []
        for sha in self.dataArrayCache:
            record = self.dataArrayCache[sha]
            array = record['array']
            count = array.GetReferenceCount()
            if count == 1 and record['ts'] < cutOffTime:
                shasToDelete.append(sha)
        for sha in shasToDelete:
            del self.dataArrayCache[sha]

    def getLastDependencyList(self, idstr):
        lastDeps = []
        if idstr in self.lastDependenciesMapping and (not self.ingoreLastDependencies):
            lastDeps = self.lastDependenciesMapping[idstr]
        return lastDeps

    def setNewDependencyList(self, idstr, depList):
        self.lastDependenciesMapping[idstr] = depList

    def buildDependencyCallList(self, idstr, newList, addMethod, removeMethod):
        oldList = self.getLastDependencyList(idstr)
        calls = []
        calls += [[addMethod, [wrapId(x)]] for x in newList if x not in oldList]
        calls += [[removeMethod, [wrapId(x)]] for x in oldList if x not in newList]
        self.setNewDependencyList(idstr, newList)
        return calls