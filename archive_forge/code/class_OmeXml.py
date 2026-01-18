from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@final
class OmeXml:
    """Create OME-TIFF XML metadata.

    Parameters:
        **metadata:
            Additional OME-XML attributes or elements to be stored.

            Creator:
                Name of creating application. The default is 'tifffile'.
            UUID:
                Unique identifier.

    Examples:
        >>> omexml = OmeXml()
        >>> omexml.addimage(
        ...     dtype='uint16',
        ...     shape=(32, 256, 256),
        ...     storedshape=(32, 1, 1, 256, 256, 1),
        ...     axes='CYX',
        ...     Name='First Image',
        ...     PhysicalSizeX=2.0,
        ... )
        >>> xml = omexml.tostring()
        >>> xml
        '<OME ...<Image ID="Image:0" Name="First Image">...</Image></OME>'
        >>> OmeXml.validate(xml)
        True

    """
    images: list[str]
    'OME-XML Image elements.'
    annotations: list[str]
    'OME-XML Annotation elements.'
    elements: list[str]
    'Other OME-XML elements.'
    _xml: str
    _ifd: int

    def __init__(self, **metadata: Any) -> None:
        metadata = metadata.get('OME', metadata)
        self._ifd = 0
        self.images = []
        self.annotations = []
        self.elements = []
        if 'UUID' in metadata:
            uuid = metadata['UUID'].split(':')[-1]
        else:
            from uuid import uuid1
            uuid = str(uuid1())
        creator = OmeXml._attribute(metadata, 'Creator', default=f'tifffile.py {__version__}')
        schema = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'
        self._xml = f'{{declaration}}<OME xmlns="{schema}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="{schema} {schema}/ome.xsd" UUID="urn:uuid:{uuid}" {creator}>{{images}}{{annotations}}{{elements}}</OME>'

    def addimage(self, dtype: DTypeLike, shape: Sequence[int], storedshape: tuple[int, int, int, int, int, int], *, axes: str | None=None, **metadata: Any) -> None:
        """Add image to OME-XML.

        The OME model can handle up to 9 dimensional images for selected
        axes orders. Refer to the OME-XML specification for details.
        Non-TZCYXS (modulo) dimensions must be after a TZC dimension or
        require an unused TZC dimension.

        Parameters:
            dtype:
                Data type of image array.
            shape:
                Shape of image array.
            storedshape:
                Normalized shape describing how image array is stored in
                TIFF file as (pages, separate_samples, depth, length, width,
                contig_samples).
            axes:
                Character codes for dimensions in `shape`.
                By default, `axes` is determined from the DimensionOrder
                metadata attribute or matched to the `shape` in reverse order
                of TZC(S)YX(S) based on `storedshape`.
                The following codes are supported: 'S' sample, 'X' width,
                'Y' length, 'Z' depth, 'C' channel, 'T' time, 'A' angle,
                'P' phase, 'R' tile, 'H' lifetime, 'E' lambda, 'Q' other.
            **metadata:
                Additional OME-XML attributes or elements to be stored.

                Image/Pixels:
                    Name, AcquisitionDate, Description, DimensionOrder,
                    PhysicalSizeX, PhysicalSizeXUnit,
                    PhysicalSizeY, PhysicalSizeYUnit,
                    PhysicalSizeZ, PhysicalSizeZUnit,
                    TimeIncrement, TimeIncrementUnit.
                Per Plane:
                    DeltaT, DeltaTUnit,
                    ExposureTime, ExposureTimeUnit,
                    PositionX, PositionXUnit,
                    PositionY, PositionYUnit,
                    PositionZ, PositionZUnit.
                Per Channel:
                    Name, AcquisitionMode, Color, ContrastMethod,
                    EmissionWavelength, EmissionWavelengthUnit,
                    ExcitationWavelength, ExcitationWavelengthUnit,
                    Fluor, IlluminationType, NDFilter,
                    PinholeSize, PinholeSizeUnit, PockelCellSetting.

        Raises:
            OmeXmlError: Image format not supported.

        """
        index = len(self.images)
        metadata = metadata.get('OME', metadata)
        metadata = metadata.get('Image', metadata)
        if isinstance(metadata, (list, tuple)):
            metadata = metadata[index]
        if 'Pixels' in metadata:
            import copy
            metadata = copy.deepcopy(metadata)
            if 'ID' in metadata['Pixels']:
                del metadata['Pixels']['ID']
            metadata.update(metadata['Pixels'])
            del metadata['Pixels']
        try:
            dtype = numpy.dtype(dtype).name
            dtype = {'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'uint8': 'uint8', 'uint16': 'uint16', 'uint32': 'uint32', 'float32': 'float', 'float64': 'double', 'complex64': 'complex', 'complex128': 'double-complex', 'bool': 'bit'}[dtype]
        except KeyError as exc:
            raise OmeXmlError(f'data type {dtype!r} not supported') from exc
        if metadata.get('Type', dtype) != dtype:
            raise OmeXmlError(f'metadata Pixels Type {metadata['Type']!r} does not match array dtype {dtype!r}')
        samples = 1
        planecount, separate, depth, length, width, contig = storedshape
        if depth != 1:
            raise OmeXmlError('ImageDepth not supported')
        if not (separate == 1 or contig == 1):
            raise ValueError('invalid stored shape')
        shape = tuple((int(i) for i in shape))
        ndim = len(shape)
        if ndim < 1 or product(shape) <= 0:
            raise OmeXmlError('empty arrays not supported')
        if axes is None:
            if contig != 1 or shape[-3:] == (length, width, 1):
                axes = 'YXS'
                samples = contig
            elif separate != 1 or (ndim == 6 and shape[-3:] == (1, length, width)):
                axes = 'SYX'
                samples = separate
            else:
                axes = 'YX'
            if not len(axes) <= ndim <= (6 if 'S' in axes else 5):
                raise OmeXmlError(f'{ndim} dimensions not supported')
            hiaxes: str = metadata.get('DimensionOrder', 'XYCZT')[:1:-1]
            axes = hiaxes[(6 if 'S' in axes else 5) - ndim:] + axes
            assert len(axes) == len(shape)
        else:
            axes = axes.upper()
            if len(axes) != len(shape):
                raise ValueError('axes do not match shape')
            if not (axes.endswith('YX') or axes.endswith('YXS') or (axes.endswith('YXC') and 'S' not in axes)):
                raise OmeXmlError('dimensions must end with YX or YXS')
            unique = []
            for ax in axes:
                if ax not in 'TZCYXSAPRHEQ':
                    raise OmeXmlError(f'dimension {ax!r} not supported')
                if ax in unique:
                    raise OmeXmlError(f'multiple {ax!r} dimensions')
                unique.append(ax)
            if ndim > (9 if 'S' in axes else 8):
                raise OmeXmlError('more than 8 dimensions not supported')
            if contig != 1:
                samples = contig
                if ndim < 3:
                    raise ValueError('dimensions do not match stored shape')
                if axes[-1] == 'C':
                    if 'S' in axes:
                        raise ValueError('invalid axes')
                    axes = axes.replace('C', 'S')
                elif axes[-1] != 'S':
                    raise ValueError('axes do not match stored shape')
                if shape[-1] != contig or shape[-2] != width:
                    raise ValueError('shape does not match stored shape')
            elif separate != 1:
                samples = separate
                if ndim < 3:
                    raise ValueError('dimensions do not match stored shape')
                if axes[-3] == 'C':
                    if 'S' in axes:
                        raise ValueError('invalid axes')
                    axes = axes.replace('C', 'S')
                elif axes[-3] != 'S':
                    raise ValueError('axes do not match stored shape')
                if shape[-3] != separate or shape[-1] != width:
                    raise ValueError('shape does not match stored shape')
        if shape[axes.index('X')] != width or shape[axes.index('Y')] != length:
            raise ValueError('shape does not match stored shape')
        if 'S' in axes:
            hiaxes = axes[:min(axes.index('S'), axes.index('Y'))]
        else:
            hiaxes = axes[:axes.index('Y')]
        if any((ax in 'APRHEQ' for ax in hiaxes)):
            modulo = {}
            dimorder = []
            axestype = {'A': 'angle', 'P': 'phase', 'R': 'tile', 'H': 'lifetime', 'E': 'lambda', 'Q': 'other'}
            for i, ax in enumerate(hiaxes):
                if ax in 'APRHEQ':
                    x = hiaxes[i - 1:i]
                    if x and x in 'TZC':
                        modulo[x] = (axestype[ax], shape[i])
                    else:
                        for x in 'TZC':
                            if x not in dimorder and x not in modulo:
                                modulo[x] = (axestype[ax], shape[i])
                                dimorder.append(x)
                                break
                        else:
                            raise OmeXmlError('more than 3 modulo dimensions')
                else:
                    dimorder.append(ax)
            hiaxes = ''.join(dimorder)
            moduloalong = ''.join((f'<ModuloAlong{ax} Type="{axtype}" Start="0" End="{size - 1}"/>' for ax, (axtype, size) in modulo.items()))
            annotationref = f'<AnnotationRef ID="Annotation:{index}"/>'
            annotations = f'<XMLAnnotation ID="Annotation:{index}" Namespace="openmicroscopy.org/omero/dimension/modulo"><Value><Modulo namespace="http://www.openmicroscopy.org/Schemas/Additions/2011-09">{moduloalong}</Modulo></Value></XMLAnnotation>'
            self.annotations.append(annotations)
        else:
            modulo = {}
            annotationref = ''
        hiaxes = hiaxes[::-1]
        for dimorder in (metadata.get('DimensionOrder', 'XYCZT'), 'XYCZT', 'XYZCT', 'XYZTC', 'XYCTZ', 'XYTCZ', 'XYTZC'):
            if hiaxes in dimorder:
                break
        else:
            raise OmeXmlError(f'dimension order {axes!r} not supported')
        dimsizes = []
        for ax in dimorder:
            if ax == 'S':
                continue
            if ax in axes:
                size = shape[axes.index(ax)]
            else:
                size = 1
            if ax == 'C':
                sizec = size
                size *= samples
            if ax in modulo:
                size *= modulo[ax][1]
            dimsizes.append(size)
        sizes = ''.join((f' Size{ax}="{size}"' for ax, size in zip(dimorder, dimsizes)))
        if 'DimensionOrder' in metadata:
            omedimorder = metadata['DimensionOrder']
            omedimorder = ''.join((ax for ax in omedimorder if dimsizes[dimorder.index(ax)] > 1))
            if hiaxes not in omedimorder:
                raise OmeXmlError(f'metadata DimensionOrder does not match {axes!r}')
        for ax, size in zip(dimorder, dimsizes):
            if metadata.get(f'Size{ax}', size) != size:
                raise OmeXmlError(f'metadata Size{ax} does not match {shape!r}')
        dimsizes[dimorder.index('C')] //= samples
        if planecount != product(dimsizes[2:]):
            raise ValueError('shape does not match stored shape')
        plane_list = []
        planeattributes = metadata.get('Plane', '')
        if planeattributes:
            cztorder = tuple((dimorder[2:].index(ax) for ax in 'CZT'))
            for p in range(planecount):
                attributes = OmeXml._attributes(planeattributes, p, 'DeltaT', 'DeltaTUnit', 'ExposureTime', 'ExposureTimeUnit', 'PositionX', 'PositionXUnit', 'PositionY', 'PositionYUnit', 'PositionZ', 'PositionZUnit')
                unraveled = numpy.unravel_index(p, dimsizes[2:], order='F')
                c, z, t = (int(unraveled[i]) for i in cztorder)
                plane_list.append(f'<Plane TheC="{c}" TheZ="{z}" TheT="{t}"{attributes}/>')
        planes = ''.join(plane_list)
        channel_list = []
        for c in range(sizec):
            lightpath = '<LightPath/>'
            attributes = OmeXml._attributes(metadata.get('Channel', ''), c, 'Name', 'AcquisitionMode', 'Color', 'ContrastMethod', 'EmissionWavelength', 'EmissionWavelengthUnit', 'ExcitationWavelength', 'ExcitationWavelengthUnit', 'Fluor', 'IlluminationType', 'NDFilter', 'PinholeSize', 'PinholeSizeUnit', 'PockelCellSetting')
            channel_list.append(f'<Channel ID="Channel:{index}:{c}" SamplesPerPixel="{samples}"{attributes}>{lightpath}</Channel>')
        channels = ''.join(channel_list)
        elements = OmeXml._elements(metadata, 'AcquisitionDate', 'Description')
        name = OmeXml._attribute(metadata, 'Name', default=f'Image{index}')
        attributes = OmeXml._attributes(metadata, None, 'SignificantBits', 'PhysicalSizeX', 'PhysicalSizeXUnit', 'PhysicalSizeY', 'PhysicalSizeYUnit', 'PhysicalSizeZ', 'PhysicalSizeZUnit', 'TimeIncrement', 'TimeIncrementUnit')
        if separate > 1 or contig > 1:
            interleaved = 'false' if separate > 1 else 'true'
            interleaved = f' Interleaved="{interleaved}"'
        else:
            interleaved = ''
        self.images.append(f'<Image ID="Image:{index}"{name}>{elements}<Pixels ID="Pixels:{index}" DimensionOrder="{dimorder}" Type="{dtype}"{sizes}{interleaved}{attributes}>{channels}<TiffData IFD="{self._ifd}" PlaneCount="{planecount}"/>{planes}</Pixels>{annotationref}</Image>')
        self._ifd += planecount

    def tostring(self, *, declaration: bool=False) -> str:
        """Return OME-XML string.

        Parameters:
            declaration: Include XML declaration.

        """
        elements = ''.join(self.elements)
        images = ''.join(self.images)
        annotations = ''.join(self.annotations)
        if annotations:
            annotations = f'<StructuredAnnotations>{annotations}</StructuredAnnotations>'
        if declaration:
            declaration_str = '<?xml version="1.0" encoding="UTF-8"?>'
        else:
            declaration_str = ''
        xml = self._xml.format(declaration=declaration_str, images=images, annotations=annotations, elements=elements)
        return xml

    def __repr__(self) -> str:
        return f'<tifffile.OmeXml @0x{id(self):016X}>'

    def __str__(self) -> str:
        """Return OME-XML string."""
        xml = self.tostring()
        try:
            from lxml import etree
            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.fromstring(xml, parser)
            xml = etree.tostring(tree, encoding='utf-8', pretty_print=True, xml_declaration=True).decode()
        except ImportError:
            pass
        except Exception as exc:
            warnings.warn(f'<tifffile.OmeXml.__str__> {exc.__class__.__name__}: {exc}', UserWarning)
        return xml

    @staticmethod
    def _escape(value: object, /) -> str:
        """Return escaped string of value."""
        if not isinstance(value, str):
            value = str(value)
        elif '&amp;' in value or '&gt;' in value or '&lt;' in value:
            return value
        value = value.replace('&', '&amp;')
        value = value.replace('>', '&gt;')
        value = value.replace('<', '&lt;')
        return value

    @staticmethod
    def _element(metadata: dict[str, Any], name: str, default: str | None=None) -> str:
        """Return XML formatted element if name in metadata."""
        value = metadata.get(name, default)
        if value is None:
            return ''
        return f'<{name}>{OmeXml._escape(value)}</{name}>'

    @staticmethod
    def _elements(metadata: dict[str, Any], /, *names: str) -> str:
        """Return XML formatted elements."""
        if not metadata:
            return ''
        elements = (OmeXml._element(metadata, name) for name in names)
        return ''.join((e for e in elements if e))

    @staticmethod
    def _attribute(metadata: dict[str, Any], name: str, /, index: int | None=None, default: Any=None) -> str:
        """Return XML formatted attribute if name in metadata."""
        value = metadata.get(name, default)
        if value is None:
            return ''
        if index is not None:
            if isinstance(value, (list, tuple)):
                try:
                    value = value[index]
                except IndexError as exc:
                    raise IndexError(f'list index out of range for attribute {name!r}') from exc
            elif index > 0:
                raise TypeError(f'{type(value).__name__!r} is not a list or tuple')
        return f' {name}="{OmeXml._escape(value)}"'

    @staticmethod
    def _attributes(metadata: dict[str, Any], index_: int | None, /, *names: str) -> str:
        """Return XML formatted attributes."""
        if not metadata:
            return ''
        if index_ is None:
            attributes = (OmeXml._attribute(metadata, name) for name in names)
        elif isinstance(metadata, (list, tuple)):
            metadata = metadata[index_]
            attributes = (OmeXml._attribute(metadata, name) for name in names)
        elif isinstance(metadata, dict):
            attributes = (OmeXml._attribute(metadata, name, index_) for name in names)
        return ''.join((a for a in attributes if a))

    @staticmethod
    def validate(omexml: str, /, omexsd: bytes | None=None, assert_: bool=True, *, _schema: list[Any]=[]) -> bool | None:
        """Return if OME-XML is valid according to XMLSchema.

        Parameters:
            omexml:
                OME-XML string to validate.
            omexsd:
                Content of OME-XSD schema to validate against.
                By default, the 2016-06 OME XMLSchema is downloaded on first
                run.
            assert\\_:
                Raise AssertionError if validation fails.
            _schema:
                Internal use.

        Raises:
            AssertionError:
                Validation failed and `assert\\_` is *True*.

        """
        from lxml import etree
        if not _schema:
            if omexsd is None:
                omexsd_path = os.path.join(os.path.dirname(__file__), 'ome.xsd')
                if os.path.exists(omexsd_path):
                    with open(omexsd_path, 'rb') as fh:
                        omexsd = fh.read()
                else:
                    import urllib.request
                    with urllib.request.urlopen('https://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd') as fh:
                        omexsd = fh.read()
            if omexsd.startswith(b'<?xml'):
                omexsd = omexsd.split(b'>', 1)[-1]
            try:
                _schema.append(etree.XMLSchema(etree.fromstring(omexsd.decode())))
            except Exception:
                _schema.append(None)
        if _schema and _schema[0] is not None:
            if omexml.startswith('<?xml'):
                omexml = omexml.split('>', 1)[-1]
            tree = etree.fromstring(omexml)
            if assert_:
                _schema[0].assert_(tree)
                return True
            return bool(_schema[0].validate(tree))
        return None