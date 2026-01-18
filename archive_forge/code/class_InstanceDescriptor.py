from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
class InstanceDescriptor(SimpleDescriptor):
    """Simple container for data related to the instance


    .. code:: python

        i2 = InstanceDescriptor()
        i2.path = instancePath2
        i2.familyName = "InstanceFamilyName"
        i2.styleName = "InstanceStyleName"
        i2.name = "instance.ufo2"
        # anisotropic location
        i2.designLocation = dict(weight=500, width=(400,300))
        i2.postScriptFontName = "InstancePostscriptName"
        i2.styleMapFamilyName = "InstanceStyleMapFamilyName"
        i2.styleMapStyleName = "InstanceStyleMapStyleName"
        i2.lib['com.coolDesignspaceApp.specimenText'] = 'Hamburgerwhatever'
        doc.addInstance(i2)
    """
    flavor = 'instance'
    _defaultLanguageCode = 'en'
    _attrs = ['filename', 'path', 'name', 'locationLabel', 'designLocation', 'userLocation', 'familyName', 'styleName', 'postScriptFontName', 'styleMapFamilyName', 'styleMapStyleName', 'localisedFamilyName', 'localisedStyleName', 'localisedStyleMapFamilyName', 'localisedStyleMapStyleName', 'glyphs', 'kerning', 'info', 'lib']
    filename = posixpath_property('_filename')
    path = posixpath_property('_path')

    def __init__(self, *, filename=None, path=None, font=None, name=None, location=None, locationLabel=None, designLocation=None, userLocation=None, familyName=None, styleName=None, postScriptFontName=None, styleMapFamilyName=None, styleMapStyleName=None, localisedFamilyName=None, localisedStyleName=None, localisedStyleMapFamilyName=None, localisedStyleMapStyleName=None, glyphs=None, kerning=True, info=True, lib=None):
        self.filename = filename
        'string. Relative path to the instance file, **as it is\n        in the document**. The file may or may not exist.\n\n        MutatorMath + VarLib.\n        '
        self.path = path
        'string. Absolute path to the instance file, calculated from\n        the document path and the string in the filename attr. The file may\n        or may not exist.\n\n        MutatorMath.\n        '
        self.font = font
        'Same as :attr:`SourceDescriptor.font`\n\n        .. seealso:: :attr:`SourceDescriptor.font`\n        '
        self.name = name
        'string. Unique identifier name of the instance, used to\n        identify it if it needs to be referenced from elsewhere in the\n        document.\n        '
        self.locationLabel = locationLabel
        'Name of a :class:`LocationLabelDescriptor`. If\n        provided, the instance should have the same location as the\n        LocationLabel.\n\n        .. seealso::\n           :meth:`getFullDesignLocation`\n           :meth:`getFullUserLocation`\n\n        .. versionadded:: 5.0\n        '
        self.designLocation: AnisotropicLocationDict = designLocation if designLocation is not None else location or {}
        'dict. Axis values for this instance, in design space coordinates.\n\n        MutatorMath + varLib.\n\n        .. seealso:: This may be only part of the full location. See:\n           :meth:`getFullDesignLocation`\n           :meth:`getFullUserLocation`\n\n        .. versionadded:: 5.0\n        '
        self.userLocation: SimpleLocationDict = userLocation or {}
        'dict. Axis values for this instance, in user space coordinates.\n\n        MutatorMath + varLib.\n\n        .. seealso:: This may be only part of the full location. See:\n           :meth:`getFullDesignLocation`\n           :meth:`getFullUserLocation`\n\n        .. versionadded:: 5.0\n        '
        self.familyName = familyName
        'string. Family name of this instance.\n\n        MutatorMath + varLib.\n        '
        self.styleName = styleName
        'string. Style name of this instance.\n\n        MutatorMath + varLib.\n        '
        self.postScriptFontName = postScriptFontName
        'string. Postscript fontname for this instance.\n\n        MutatorMath + varLib.\n        '
        self.styleMapFamilyName = styleMapFamilyName
        'string. StyleMap familyname for this instance.\n\n        MutatorMath + varLib.\n        '
        self.styleMapStyleName = styleMapStyleName
        'string. StyleMap stylename for this instance.\n\n        MutatorMath + varLib.\n        '
        self.localisedFamilyName = localisedFamilyName or {}
        'dict. A dictionary of localised family name\n        strings, keyed by language code.\n        '
        self.localisedStyleName = localisedStyleName or {}
        'dict. A dictionary of localised stylename\n        strings, keyed by language code.\n        '
        self.localisedStyleMapFamilyName = localisedStyleMapFamilyName or {}
        'A dictionary of localised style map\n        familyname strings, keyed by language code.\n        '
        self.localisedStyleMapStyleName = localisedStyleMapStyleName or {}
        'A dictionary of localised style map\n        stylename strings, keyed by language code.\n        '
        self.glyphs = glyphs or {}
        'dict for special master definitions for glyphs. If glyphs\n        need special masters (to record the results of executed rules for\n        example).\n\n        MutatorMath.\n\n        .. deprecated:: 5.0\n            Use rules or sparse sources instead.\n        '
        self.kerning = kerning
        ' bool. Indicates if this instance needs its kerning\n        calculated.\n\n        MutatorMath.\n\n        .. deprecated:: 5.0\n        '
        self.info = info
        'bool. Indicated if this instance needs the interpolating\n        font.info calculated.\n\n        .. deprecated:: 5.0\n        '
        self.lib = lib or {}
        'Custom data associated with this instance.'

    @property
    def location(self):
        """dict. Axis values for this instance.

        MutatorMath + varLib.

        .. deprecated:: 5.0
           Use the more explicit alias for this property :attr:`designLocation`.
        """
        return self.designLocation

    @location.setter
    def location(self, location: Optional[AnisotropicLocationDict]):
        self.designLocation = location or {}

    def setStyleName(self, styleName, languageCode='en'):
        """These methods give easier access to the localised names."""
        self.localisedStyleName[languageCode] = tostr(styleName)

    def getStyleName(self, languageCode='en'):
        return self.localisedStyleName.get(languageCode)

    def setFamilyName(self, familyName, languageCode='en'):
        self.localisedFamilyName[languageCode] = tostr(familyName)

    def getFamilyName(self, languageCode='en'):
        return self.localisedFamilyName.get(languageCode)

    def setStyleMapStyleName(self, styleMapStyleName, languageCode='en'):
        self.localisedStyleMapStyleName[languageCode] = tostr(styleMapStyleName)

    def getStyleMapStyleName(self, languageCode='en'):
        return self.localisedStyleMapStyleName.get(languageCode)

    def setStyleMapFamilyName(self, styleMapFamilyName, languageCode='en'):
        self.localisedStyleMapFamilyName[languageCode] = tostr(styleMapFamilyName)

    def getStyleMapFamilyName(self, languageCode='en'):
        return self.localisedStyleMapFamilyName.get(languageCode)

    def clearLocation(self, axisName: Optional[str]=None):
        """Clear all location-related fields. Ensures that
        :attr:``designLocation`` and :attr:``userLocation`` are dictionaries
        (possibly empty if clearing everything).

        In order to update the location of this instance wholesale, a user
        should first clear all the fields, then change the field(s) for which
        they have data.

        .. code:: python

            instance.clearLocation()
            instance.designLocation = {'Weight': (34, 36.5), 'Width': 100}
            instance.userLocation = {'Opsz': 16}

        In order to update a single axis location, the user should only clear
        that axis, then edit the values:

        .. code:: python

            instance.clearLocation('Weight')
            instance.designLocation['Weight'] = (34, 36.5)

        Args:
          axisName: if provided, only clear the location for that axis.

        .. versionadded:: 5.0
        """
        self.locationLabel = None
        if axisName is None:
            self.designLocation = {}
            self.userLocation = {}
        else:
            if self.designLocation is None:
                self.designLocation = {}
            if axisName in self.designLocation:
                del self.designLocation[axisName]
            if self.userLocation is None:
                self.userLocation = {}
            if axisName in self.userLocation:
                del self.userLocation[axisName]

    def getLocationLabelDescriptor(self, doc: 'DesignSpaceDocument') -> Optional[LocationLabelDescriptor]:
        """Get the :class:`LocationLabelDescriptor` instance that matches
        this instances's :attr:`locationLabel`.

        Raises if the named label can't be found.

        .. versionadded:: 5.0
        """
        if self.locationLabel is None:
            return None
        label = doc.getLocationLabel(self.locationLabel)
        if label is None:
            raise DesignSpaceDocumentError(f'InstanceDescriptor.getLocationLabelDescriptor(): unknown location label `{self.locationLabel}` in instance `{self.name}`.')
        return label

    def getFullDesignLocation(self, doc: 'DesignSpaceDocument') -> AnisotropicLocationDict:
        """Get the complete design location of this instance, by combining data
        from the various location fields, default axis values and mappings, and
        top-level location labels.

        The source of truth for this instance's location is determined for each
        axis independently by taking the first not-None field in this list:

        - ``locationLabel``: the location along this axis is the same as the
          matching STAT format 4 label. No anisotropy.
        - ``designLocation[axisName]``: the explicit design location along this
          axis, possibly anisotropic.
        - ``userLocation[axisName]``: the explicit user location along this
          axis. No anisotropy.
        - ``axis.default``: default axis value. No anisotropy.

        .. versionadded:: 5.0
        """
        label = self.getLocationLabelDescriptor(doc)
        if label is not None:
            return doc.map_forward(label.userLocation)
        result: AnisotropicLocationDict = {}
        for axis in doc.axes:
            if axis.name in self.designLocation:
                result[axis.name] = self.designLocation[axis.name]
            elif axis.name in self.userLocation:
                result[axis.name] = axis.map_forward(self.userLocation[axis.name])
            else:
                result[axis.name] = axis.map_forward(axis.default)
        return result

    def getFullUserLocation(self, doc: 'DesignSpaceDocument') -> SimpleLocationDict:
        """Get the complete user location for this instance.

        .. seealso:: :meth:`getFullDesignLocation`

        .. versionadded:: 5.0
        """
        return doc.map_backward(self.getFullDesignLocation(doc))