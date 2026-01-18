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
class _TIFF:
    """Delay-loaded constants, accessible via :py:attr:`TIFF` instance."""

    @cached_property
    def CLASSIC_LE(self) -> TiffFormat:
        """32-bit little-endian TIFF format."""
        return TiffFormat(version=42, byteorder='<', offsetsize=4, offsetformat='<I', tagnosize=2, tagnoformat='<H', tagsize=12, tagformat1='<HH', tagformat2='<I4s', tagoffsetthreshold=4)

    @cached_property
    def CLASSIC_BE(self) -> TiffFormat:
        """32-bit big-endian TIFF format."""
        return TiffFormat(version=42, byteorder='>', offsetsize=4, offsetformat='>I', tagnosize=2, tagnoformat='>H', tagsize=12, tagformat1='>HH', tagformat2='>I4s', tagoffsetthreshold=4)

    @cached_property
    def BIG_LE(self) -> TiffFormat:
        """64-bit little-endian TIFF format."""
        return TiffFormat(version=43, byteorder='<', offsetsize=8, offsetformat='<Q', tagnosize=8, tagnoformat='<Q', tagsize=20, tagformat1='<HH', tagformat2='<Q8s', tagoffsetthreshold=8)

    @cached_property
    def BIG_BE(self) -> TiffFormat:
        """64-bit big-endian TIFF format."""
        return TiffFormat(version=43, byteorder='>', offsetsize=8, offsetformat='>Q', tagnosize=8, tagnoformat='>Q', tagsize=20, tagformat1='>HH', tagformat2='>Q8s', tagoffsetthreshold=8)

    @cached_property
    def NDPI_LE(self) -> TiffFormat:
        """32-bit little-endian TIFF format with 64-bit offsets."""
        return TiffFormat(version=42, byteorder='<', offsetsize=8, offsetformat='<Q', tagnosize=2, tagnoformat='<H', tagsize=12, tagformat1='<HH', tagformat2='<I8s', tagoffsetthreshold=4)

    @cached_property
    def TAGS(self) -> TiffTagRegistry:
        """Registry of TIFF tag codes and names from TIFF6, TIFF/EP, EXIF."""
        return TiffTagRegistry(((11, 'ProcessingSoftware'), (254, 'NewSubfileType'), (255, 'SubfileType'), (256, 'ImageWidth'), (257, 'ImageLength'), (258, 'BitsPerSample'), (259, 'Compression'), (262, 'PhotometricInterpretation'), (263, 'Thresholding'), (264, 'CellWidth'), (265, 'CellLength'), (266, 'FillOrder'), (269, 'DocumentName'), (270, 'ImageDescription'), (271, 'Make'), (272, 'Model'), (273, 'StripOffsets'), (274, 'Orientation'), (277, 'SamplesPerPixel'), (278, 'RowsPerStrip'), (279, 'StripByteCounts'), (280, 'MinSampleValue'), (281, 'MaxSampleValue'), (282, 'XResolution'), (283, 'YResolution'), (284, 'PlanarConfiguration'), (285, 'PageName'), (286, 'XPosition'), (287, 'YPosition'), (288, 'FreeOffsets'), (289, 'FreeByteCounts'), (290, 'GrayResponseUnit'), (291, 'GrayResponseCurve'), (292, 'T4Options'), (293, 'T6Options'), (296, 'ResolutionUnit'), (297, 'PageNumber'), (300, 'ColorResponseUnit'), (301, 'TransferFunction'), (305, 'Software'), (306, 'DateTime'), (315, 'Artist'), (316, 'HostComputer'), (317, 'Predictor'), (318, 'WhitePoint'), (319, 'PrimaryChromaticities'), (320, 'ColorMap'), (321, 'HalftoneHints'), (322, 'TileWidth'), (323, 'TileLength'), (324, 'TileOffsets'), (325, 'TileByteCounts'), (326, 'BadFaxLines'), (327, 'CleanFaxData'), (328, 'ConsecutiveBadFaxLines'), (330, 'SubIFDs'), (332, 'InkSet'), (333, 'InkNames'), (334, 'NumberOfInks'), (336, 'DotRange'), (337, 'TargetPrinter'), (338, 'ExtraSamples'), (339, 'SampleFormat'), (340, 'SMinSampleValue'), (341, 'SMaxSampleValue'), (342, 'TransferRange'), (343, 'ClipPath'), (344, 'XClipPathUnits'), (345, 'YClipPathUnits'), (346, 'Indexed'), (347, 'JPEGTables'), (351, 'OPIProxy'), (400, 'GlobalParametersIFD'), (401, 'ProfileType'), (402, 'FaxProfile'), (403, 'CodingMethods'), (404, 'VersionYear'), (405, 'ModeNumber'), (433, 'Decode'), (434, 'DefaultImageColor'), (435, 'T82Options'), (437, 'JPEGTables'), (512, 'JPEGProc'), (513, 'JPEGInterchangeFormat'), (514, 'JPEGInterchangeFormatLength'), (515, 'JPEGRestartInterval'), (517, 'JPEGLosslessPredictors'), (518, 'JPEGPointTransforms'), (519, 'JPEGQTables'), (520, 'JPEGDCTables'), (521, 'JPEGACTables'), (529, 'YCbCrCoefficients'), (530, 'YCbCrSubSampling'), (531, 'YCbCrPositioning'), (532, 'ReferenceBlackWhite'), (559, 'StripRowCounts'), (700, 'XMP'), (769, 'GDIGamma'), (770, 'ICCProfileDescriptor'), (771, 'SRGBRenderingIntent'), (800, 'ImageTitle'), (907, 'SiffCompress'), (999, 'USPTO_Miscellaneous'), (4864, 'AndorId'), (4869, 'AndorTemperature'), (4876, 'AndorExposureTime'), (4878, 'AndorKineticCycleTime'), (4879, 'AndorAccumulations'), (4881, 'AndorAcquisitionCycleTime'), (4882, 'AndorReadoutTime'), (4884, 'AndorPhotonCounting'), (4885, 'AndorEmDacLevel'), (4890, 'AndorFrames'), (4896, 'AndorHorizontalFlip'), (4897, 'AndorVerticalFlip'), (4898, 'AndorClockwise'), (4899, 'AndorCounterClockwise'), (4904, 'AndorVerticalClockVoltage'), (4905, 'AndorVerticalShiftSpeed'), (4907, 'AndorPreAmpSetting'), (4908, 'AndorCameraSerial'), (4911, 'AndorActualTemperature'), (4912, 'AndorBaselineClamp'), (4913, 'AndorPrescans'), (4914, 'AndorModel'), (4915, 'AndorChipSizeX'), (4916, 'AndorChipSizeY'), (4944, 'AndorBaselineOffset'), (4966, 'AndorSoftwareVersion'), (18246, 'Rating'), (18247, 'XP_DIP_XML'), (18248, 'StitchInfo'), (18249, 'RatingPercent'), (20481, 'ResolutionXUnit'), (20482, 'ResolutionYUnit'), (20483, 'ResolutionXLengthUnit'), (20484, 'ResolutionYLengthUnit'), (20485, 'PrintFlags'), (20486, 'PrintFlagsVersion'), (20487, 'PrintFlagsCrop'), (20488, 'PrintFlagsBleedWidth'), (20489, 'PrintFlagsBleedWidthScale'), (20490, 'HalftoneLPI'), (20491, 'HalftoneLPIUnit'), (20492, 'HalftoneDegree'), (20493, 'HalftoneShape'), (20494, 'HalftoneMisc'), (20495, 'HalftoneScreen'), (20496, 'JPEGQuality'), (20497, 'GridSize'), (20498, 'ThumbnailFormat'), (20499, 'ThumbnailWidth'), (20500, 'ThumbnailHeight'), (20501, 'ThumbnailColorDepth'), (20502, 'ThumbnailPlanes'), (20503, 'ThumbnailRawBytes'), (20504, 'ThumbnailSize'), (20505, 'ThumbnailCompressedSize'), (20506, 'ColorTransferFunction'), (20507, 'ThumbnailData'), (20512, 'ThumbnailImageWidth'), (20513, 'ThumbnailImageHeight'), (20514, 'ThumbnailBitsPerSample'), (20515, 'ThumbnailCompression'), (20516, 'ThumbnailPhotometricInterp'), (20517, 'ThumbnailImageDescription'), (20518, 'ThumbnailEquipMake'), (20519, 'ThumbnailEquipModel'), (20520, 'ThumbnailStripOffsets'), (20521, 'ThumbnailOrientation'), (20522, 'ThumbnailSamplesPerPixel'), (20523, 'ThumbnailRowsPerStrip'), (20524, 'ThumbnailStripBytesCount'), (20525, 'ThumbnailResolutionX'), (20526, 'ThumbnailResolutionY'), (20527, 'ThumbnailPlanarConfig'), (20528, 'ThumbnailResolutionUnit'), (20529, 'ThumbnailTransferFunction'), (20530, 'ThumbnailSoftwareUsed'), (20531, 'ThumbnailDateTime'), (20532, 'ThumbnailArtist'), (20533, 'ThumbnailWhitePoint'), (20534, 'ThumbnailPrimaryChromaticities'), (20535, 'ThumbnailYCbCrCoefficients'), (20536, 'ThumbnailYCbCrSubsampling'), (20537, 'ThumbnailYCbCrPositioning'), (20538, 'ThumbnailRefBlackWhite'), (20539, 'ThumbnailCopyRight'), (20545, 'InteroperabilityIndex'), (20546, 'InteroperabilityVersion'), (20624, 'LuminanceTable'), (20625, 'ChrominanceTable'), (20736, 'FrameDelay'), (20737, 'LoopCount'), (20738, 'GlobalPalette'), (20739, 'IndexBackground'), (20740, 'IndexTransparent'), (20752, 'PixelUnit'), (20753, 'PixelPerUnitX'), (20754, 'PixelPerUnitY'), (20755, 'PaletteHistogram'), (28672, 'SonyRawFileType'), (28722, 'VignettingCorrParams'), (28725, 'ChromaticAberrationCorrParams'), (28727, 'DistortionCorrParams'), (32781, 'ImageID'), (32931, 'WangTag1'), (32932, 'WangAnnotation'), (32933, 'WangTag3'), (32934, 'WangTag4'), (32953, 'ImageReferencePoints'), (32954, 'RegionXformTackPoint'), (32955, 'WarpQuadrilateral'), (32956, 'AffineTransformMat'), (32995, 'Matteing'), (32996, 'DataType'), (32997, 'ImageDepth'), (32998, 'TileDepth'), (33300, 'ImageFullWidth'), (33301, 'ImageFullLength'), (33302, 'TextureFormat'), (33303, 'TextureWrapModes'), (33304, 'FieldOfViewCotangent'), (33305, 'MatrixWorldToScreen'), (33306, 'MatrixWorldToCamera'), (33405, 'Model2'), (33421, 'CFARepeatPatternDim'), (33422, 'CFAPattern'), (33423, 'BatteryLevel'), (33424, 'KodakIFD'), (33434, 'ExposureTime'), (33437, 'FNumber'), (33432, 'Copyright'), (33445, 'MDFileTag'), (33446, 'MDScalePixel'), (33447, 'MDColorTable'), (33448, 'MDLabName'), (33449, 'MDSampleInfo'), (33450, 'MDPrepDate'), (33451, 'MDPrepTime'), (33452, 'MDFileUnits'), (33465, 'NiffRotation'), (33466, 'NiffNavyCompression'), (33467, 'NiffTileIndex'), (33471, 'OlympusINI'), (33550, 'ModelPixelScaleTag'), (33560, 'OlympusSIS'), (33589, 'AdventScale'), (33590, 'AdventRevision'), (33628, 'UIC1tag'), (33629, 'UIC2tag'), (33630, 'UIC3tag'), (33631, 'UIC4tag'), (33723, 'IPTCNAA'), (33858, 'ExtendedTagsOffset'), (33918, 'IntergraphPacketData'), (33919, 'IntergraphFlagRegisters'), (33920, 'IntergraphMatrixTag'), (33921, 'INGRReserved'), (33922, 'ModelTiepointTag'), (33923, 'LeicaMagic'), (34016, 'Site'), (34017, 'ColorSequence'), (34018, 'IT8Header'), (34019, 'RasterPadding'), (34020, 'BitsPerRunLength'), (34021, 'BitsPerExtendedRunLength'), (34022, 'ColorTable'), (34023, 'ImageColorIndicator'), (34024, 'BackgroundColorIndicator'), (34025, 'ImageColorValue'), (34026, 'BackgroundColorValue'), (34027, 'PixelIntensityRange'), (34028, 'TransparencyIndicator'), (34029, 'ColorCharacterization'), (34030, 'HCUsage'), (34031, 'TrapIndicator'), (34032, 'CMYKEquivalent'), (34118, 'CZ_SEM'), (34152, 'AFCP_IPTC'), (34232, 'PixelMagicJBIGOptions'), (34263, 'JPLCartoIFD'), (34122, 'IPLAB'), (34264, 'ModelTransformationTag'), (34306, 'WB_GRGBLevels'), (34310, 'LeafData'), (34361, 'MM_Header'), (34362, 'MM_Stamp'), (34363, 'MM_Unknown'), (34377, 'ImageResources'), (34386, 'MM_UserBlock'), (34412, 'CZ_LSMINFO'), (34665, 'ExifTag'), (34675, 'InterColorProfile'), (34680, 'FEI_SFEG'), (34682, 'FEI_HELIOS'), (34683, 'FEI_TITAN'), (34687, 'FXExtensions'), (34688, 'MultiProfiles'), (34689, 'SharedData'), (34690, 'T88Options'), (34710, 'MarCCD'), (34732, 'ImageLayer'), (34735, 'GeoKeyDirectoryTag'), (34736, 'GeoDoubleParamsTag'), (34737, 'GeoAsciiParamsTag'), (34750, 'JBIGOptions'), (34821, 'PIXTIFF'), (34850, 'ExposureProgram'), (34852, 'SpectralSensitivity'), (34853, 'GPSTag'), (34853, 'OlympusSIS2'), (34855, 'ISOSpeedRatings'), (34855, 'PhotographicSensitivity'), (34856, 'OECF'), (34857, 'Interlace'), (34858, 'TimeZoneOffset'), (34859, 'SelfTimerMode'), (34864, 'SensitivityType'), (34865, 'StandardOutputSensitivity'), (34866, 'RecommendedExposureIndex'), (34867, 'ISOSpeed'), (34868, 'ISOSpeedLatitudeyyy'), (34869, 'ISOSpeedLatitudezzz'), (34908, 'HylaFAXFaxRecvParams'), (34909, 'HylaFAXFaxSubAddress'), (34910, 'HylaFAXFaxRecvTime'), (34911, 'FaxDcs'), (34929, 'FedexEDR'), (34954, 'LeafSubIFD'), (34959, 'Aphelion1'), (34960, 'Aphelion2'), (34961, 'AphelionInternal'), (36864, 'ExifVersion'), (36867, 'DateTimeOriginal'), (36868, 'DateTimeDigitized'), (36873, 'GooglePlusUploadCode'), (36880, 'OffsetTime'), (36881, 'OffsetTimeOriginal'), (36882, 'OffsetTimeDigitized'), (36864, 'TVX_Unknown'), (36865, 'TVX_NumExposure'), (36866, 'TVX_NumBackground'), (36867, 'TVX_ExposureTime'), (36868, 'TVX_BackgroundTime'), (36870, 'TVX_Unknown'), (36873, 'TVX_SubBpp'), (36874, 'TVX_SubWide'), (36875, 'TVX_SubHigh'), (36876, 'TVX_BlackLevel'), (36877, 'TVX_DarkCurrent'), (36878, 'TVX_ReadNoise'), (36879, 'TVX_DarkCurrentNoise'), (36880, 'TVX_BeamMonitor'), (37120, 'TVX_UserVariables'), (37121, 'ComponentsConfiguration'), (37122, 'CompressedBitsPerPixel'), (37377, 'ShutterSpeedValue'), (37378, 'ApertureValue'), (37379, 'BrightnessValue'), (37380, 'ExposureBiasValue'), (37381, 'MaxApertureValue'), (37382, 'SubjectDistance'), (37383, 'MeteringMode'), (37384, 'LightSource'), (37385, 'Flash'), (37386, 'FocalLength'), (37387, 'FlashEnergy'), (37388, 'SpatialFrequencyResponse'), (37389, 'Noise'), (37390, 'FocalPlaneXResolution'), (37391, 'FocalPlaneYResolution'), (37392, 'FocalPlaneResolutionUnit'), (37393, 'ImageNumber'), (37394, 'SecurityClassification'), (37395, 'ImageHistory'), (37396, 'SubjectLocation'), (37397, 'ExposureIndex'), (37398, 'TIFFEPStandardID'), (37399, 'SensingMethod'), (37434, 'CIP3DataFile'), (37435, 'CIP3Sheet'), (37436, 'CIP3Side'), (37439, 'StoNits'), (37500, 'MakerNote'), (37510, 'UserComment'), (37520, 'SubsecTime'), (37521, 'SubsecTimeOriginal'), (37522, 'SubsecTimeDigitized'), (37679, 'MODIText'), (37680, 'MODIOLEPropertySetStorage'), (37681, 'MODIPositioning'), (37701, 'AgilentBinary'), (37702, 'AgilentString'), (37706, 'TVIPS'), (37707, 'TVIPS1'), (37708, 'TVIPS2'), (37724, 'ImageSourceData'), (37888, 'Temperature'), (37889, 'Humidity'), (37890, 'Pressure'), (37891, 'WaterDepth'), (37892, 'Acceleration'), (37893, 'CameraElevationAngle'), (40000, 'XPos'), (40001, 'YPos'), (40002, 'ZPos'), (40001, 'MC_IpWinScal'), (40001, 'RecipName'), (40002, 'RecipNumber'), (40003, 'SenderName'), (40004, 'Routing'), (40005, 'CallerId'), (40006, 'TSID'), (40007, 'CSID'), (40008, 'FaxTime'), (40100, 'MC_IdOld'), (40106, 'MC_Unknown'), (40965, 'InteroperabilityTag'), (40091, 'XPTitle'), (40092, 'XPComment'), (40093, 'XPAuthor'), (40094, 'XPKeywords'), (40095, 'XPSubject'), (40960, 'FlashpixVersion'), (40961, 'ColorSpace'), (40962, 'PixelXDimension'), (40963, 'PixelYDimension'), (40964, 'RelatedSoundFile'), (40976, 'SamsungRawPointersOffset'), (40977, 'SamsungRawPointersLength'), (41217, 'SamsungRawByteOrder'), (41218, 'SamsungRawUnknown'), (41483, 'FlashEnergy'), (41484, 'SpatialFrequencyResponse'), (41485, 'Noise'), (41486, 'FocalPlaneXResolution'), (41487, 'FocalPlaneYResolution'), (41488, 'FocalPlaneResolutionUnit'), (41489, 'ImageNumber'), (41490, 'SecurityClassification'), (41491, 'ImageHistory'), (41492, 'SubjectLocation'), (41493, 'ExposureIndex '), (41494, 'TIFF-EPStandardID'), (41495, 'SensingMethod'), (41728, 'FileSource'), (41729, 'SceneType'), (41730, 'CFAPattern'), (41985, 'CustomRendered'), (41986, 'ExposureMode'), (41987, 'WhiteBalance'), (41988, 'DigitalZoomRatio'), (41989, 'FocalLengthIn35mmFilm'), (41990, 'SceneCaptureType'), (41991, 'GainControl'), (41992, 'Contrast'), (41993, 'Saturation'), (41994, 'Sharpness'), (41995, 'DeviceSettingDescription'), (41996, 'SubjectDistanceRange'), (42016, 'ImageUniqueID'), (42032, 'CameraOwnerName'), (42033, 'BodySerialNumber'), (42034, 'LensSpecification'), (42035, 'LensMake'), (42036, 'LensModel'), (42037, 'LensSerialNumber'), (42080, 'CompositeImage'), (42081, 'SourceImageNumberCompositeImage'), (42082, 'SourceExposureTimesCompositeImage'), (42112, 'GDAL_METADATA'), (42113, 'GDAL_NODATA'), (42240, 'Gamma'), (43314, 'NIHImageHeader'), (44992, 'ExpandSoftware'), (44993, 'ExpandLens'), (44994, 'ExpandFilm'), (44995, 'ExpandFilterLens'), (44996, 'ExpandScanner'), (44997, 'ExpandFlashLamp'), (48129, 'PixelFormat'), (48130, 'Transformation'), (48131, 'Uncompressed'), (48132, 'ImageType'), (48256, 'ImageWidth'), (48257, 'ImageHeight'), (48258, 'WidthResolution'), (48259, 'HeightResolution'), (48320, 'ImageOffset'), (48321, 'ImageByteCount'), (48322, 'AlphaOffset'), (48323, 'AlphaByteCount'), (48324, 'ImageDataDiscard'), (48325, 'AlphaDataDiscard'), (50003, 'KodakAPP3'), (50215, 'OceScanjobDescription'), (50216, 'OceApplicationSelector'), (50217, 'OceIdentificationNumber'), (50218, 'OceImageLogicCharacteristics'), (50255, 'Annotations'), (50288, 'MC_Id'), (50289, 'MC_XYPosition'), (50290, 'MC_ZPosition'), (50291, 'MC_XYCalibration'), (50292, 'MC_LensCharacteristics'), (50293, 'MC_ChannelName'), (50294, 'MC_ExcitationWavelength'), (50295, 'MC_TimeStamp'), (50296, 'MC_FrameProperties'), (50341, 'PrintImageMatching'), (50495, 'PCO_RAW'), (50547, 'OriginalFileName'), (50560, 'USPTO_OriginalContentType'), (50561, 'USPTO_RotationCode'), (50648, 'CR2Unknown1'), (50649, 'CR2Unknown2'), (50656, 'CR2CFAPattern'), (50674, 'LercParameters'), (50706, 'DNGVersion'), (50707, 'DNGBackwardVersion'), (50708, 'UniqueCameraModel'), (50709, 'LocalizedCameraModel'), (50710, 'CFAPlaneColor'), (50711, 'CFALayout'), (50712, 'LinearizationTable'), (50713, 'BlackLevelRepeatDim'), (50714, 'BlackLevel'), (50715, 'BlackLevelDeltaH'), (50716, 'BlackLevelDeltaV'), (50717, 'WhiteLevel'), (50718, 'DefaultScale'), (50719, 'DefaultCropOrigin'), (50720, 'DefaultCropSize'), (50721, 'ColorMatrix1'), (50722, 'ColorMatrix2'), (50723, 'CameraCalibration1'), (50724, 'CameraCalibration2'), (50725, 'ReductionMatrix1'), (50726, 'ReductionMatrix2'), (50727, 'AnalogBalance'), (50728, 'AsShotNeutral'), (50729, 'AsShotWhiteXY'), (50730, 'BaselineExposure'), (50731, 'BaselineNoise'), (50732, 'BaselineSharpness'), (50733, 'BayerGreenSplit'), (50734, 'LinearResponseLimit'), (50735, 'CameraSerialNumber'), (50736, 'LensInfo'), (50737, 'ChromaBlurRadius'), (50738, 'AntiAliasStrength'), (50739, 'ShadowScale'), (50740, 'DNGPrivateData'), (50741, 'MakerNoteSafety'), (50752, 'RawImageSegmentation'), (50778, 'CalibrationIlluminant1'), (50779, 'CalibrationIlluminant2'), (50780, 'BestQualityScale'), (50781, 'RawDataUniqueID'), (50784, 'AliasLayerMetadata'), (50827, 'OriginalRawFileName'), (50828, 'OriginalRawFileData'), (50829, 'ActiveArea'), (50830, 'MaskedAreas'), (50831, 'AsShotICCProfile'), (50832, 'AsShotPreProfileMatrix'), (50833, 'CurrentICCProfile'), (50834, 'CurrentPreProfileMatrix'), (50838, 'IJMetadataByteCounts'), (50839, 'IJMetadata'), (50844, 'RPCCoefficientTag'), (50879, 'ColorimetricReference'), (50885, 'SRawType'), (50898, 'PanasonicTitle'), (50899, 'PanasonicTitle2'), (50908, 'RSID'), (50909, 'GEO_METADATA'), (50931, 'CameraCalibrationSignature'), (50932, 'ProfileCalibrationSignature'), (50933, 'ProfileIFD'), (50934, 'AsShotProfileName'), (50935, 'NoiseReductionApplied'), (50936, 'ProfileName'), (50937, 'ProfileHueSatMapDims'), (50938, 'ProfileHueSatMapData1'), (50939, 'ProfileHueSatMapData2'), (50940, 'ProfileToneCurve'), (50941, 'ProfileEmbedPolicy'), (50942, 'ProfileCopyright'), (50964, 'ForwardMatrix1'), (50965, 'ForwardMatrix2'), (50966, 'PreviewApplicationName'), (50967, 'PreviewApplicationVersion'), (50968, 'PreviewSettingsName'), (50969, 'PreviewSettingsDigest'), (50970, 'PreviewColorSpace'), (50971, 'PreviewDateTime'), (50972, 'RawImageDigest'), (50973, 'OriginalRawFileDigest'), (50974, 'SubTileBlockSize'), (50975, 'RowInterleaveFactor'), (50981, 'ProfileLookTableDims'), (50982, 'ProfileLookTableData'), (51008, 'OpcodeList1'), (51009, 'OpcodeList2'), (51022, 'OpcodeList3'), (51023, 'FibicsXML'), (51041, 'NoiseProfile'), (51043, 'TimeCodes'), (51044, 'FrameRate'), (51058, 'TStop'), (51081, 'ReelName'), (51089, 'OriginalDefaultFinalSize'), (51090, 'OriginalBestQualitySize'), (51091, 'OriginalDefaultCropSize'), (51105, 'CameraLabel'), (51107, 'ProfileHueSatMapEncoding'), (51108, 'ProfileLookTableEncoding'), (51109, 'BaselineExposureOffset'), (51110, 'DefaultBlackRender'), (51111, 'NewRawImageDigest'), (51112, 'RawToPreviewGain'), (51113, 'CacheBlob'), (51114, 'CacheVersion'), (51123, 'MicroManagerMetadata'), (51125, 'DefaultUserCrop'), (51159, 'ZIFmetadata'), (51160, 'ZIFannotations'), (51177, 'DepthFormat'), (51178, 'DepthNear'), (51179, 'DepthFar'), (51180, 'DepthUnits'), (51181, 'DepthMeasureType'), (51182, 'EnhanceParams'), (52525, 'ProfileGainTableMap'), (52526, 'SemanticName'), (52528, 'SemanticInstanceID'), (52536, 'MaskSubArea'), (52543, 'RGBTables'), (52529, 'CalibrationIlluminant3'), (52531, 'ColorMatrix3'), (52530, 'CameraCalibration3'), (52538, 'ReductionMatrix3'), (52537, 'ProfileHueSatMapData3'), (52532, 'ForwardMatrix3'), (52533, 'IlluminantData1'), (52534, 'IlluminantData2'), (53535, 'IlluminantData3'), (55000, 'AperioUnknown55000'), (55001, 'AperioMagnification'), (55002, 'AperioMPP'), (55003, 'AperioScanScopeID'), (55004, 'AperioDate'), (59932, 'Padding'), (59933, 'OffsetSchema'), (65200, 'FlexXML')))

    @cached_property
    def TAG_READERS(self) -> dict[int, Callable[[FileHandle, ByteOrder, int, int, int], Any]]:
        return {301: read_colormap, 320: read_colormap, 33723: read_bytes, 33628: read_uic1tag, 33629: read_uic2tag, 33630: read_uic3tag, 33631: read_uic4tag, 34118: read_cz_sem, 34361: read_mm_header, 34362: read_mm_stamp, 34363: read_numpy, 34386: read_numpy, 34412: read_cz_lsminfo, 34680: read_fei_metadata, 34682: read_fei_metadata, 37706: read_tvips_header, 37724: read_bytes, 33923: read_bytes, 43314: read_nih_image_header, 40100: read_bytes, 50288: read_bytes, 50296: read_bytes, 50839: read_bytes, 51123: read_json, 33471: read_sis_ini, 33560: read_sis, 34665: read_exif_ifd, 34853: read_gps_ifd, 40965: read_interoperability_ifd, 65426: read_numpy, 65432: read_numpy, 65439: read_numpy, 65459: read_bytes}

    @cached_property
    def TAG_LOAD(self) -> frozenset[int]:
        return frozenset((258, 270, 273, 277, 279, 282, 283, 305, 324, 325, 330, 338, 339, 347, 513, 514, 530, 33628, 42113, 50838, 50839))

    @cached_property
    def TAG_FILTERED(self) -> frozenset[int]:
        return frozenset((256, 257, 258, 259, 262, 266, 273, 277, 278, 279, 284, 317, 322, 323, 324, 325, 330, 338, 339, 400, 32997, 32998, 34665, 34853, 40965))

    @cached_property
    def TAG_TUPLE(self) -> frozenset[int]:
        return frozenset((273, 279, 282, 283, 324, 325, 330, 338, 513, 514, 530, 531, 34736, 50838))

    @cached_property
    def TAG_ATTRIBUTES(self) -> dict[int, str]:
        return {254: 'subfiletype', 256: 'imagewidth', 257: 'imagelength', 259: 'compression', 262: 'photometric', 266: 'fillorder', 270: 'description', 277: 'samplesperpixel', 278: 'rowsperstrip', 284: 'planarconfig', 305: 'software', 317: 'predictor', 322: 'tilewidth', 323: 'tilelength', 330: 'subifds', 338: 'extrasamples', 347: 'jpegtables', 530: 'subsampling', 32997: 'imagedepth', 32998: 'tiledepth'}

    @cached_property
    def TAG_ENUM(self) -> dict[int, type[enum.Enum]]:
        return {254: FILETYPE, 255: OFILETYPE, 259: COMPRESSION, 262: PHOTOMETRIC, 266: FILLORDER, 274: ORIENTATION, 284: PLANARCONFIG, 296: RESUNIT, 317: PREDICTOR, 338: EXTRASAMPLE, 339: SAMPLEFORMAT}

    @cached_property
    def EXIF_TAGS(self) -> TiffTagRegistry:
        """Registry of EXIF tags, including private Photoshop Camera RAW."""
        tags = TiffTagRegistry(((65000, 'OwnerName'), (65001, 'SerialNumber'), (65002, 'Lens'), (65100, 'RawFile'), (65101, 'Converter'), (65102, 'WhiteBalance'), (65105, 'Exposure'), (65106, 'Shadows'), (65107, 'Brightness'), (65108, 'Contrast'), (65109, 'Saturation'), (65110, 'Sharpness'), (65111, 'Smoothness'), (65112, 'MoireFilter')))
        tags.update(TIFF.TAGS)
        return tags

    @cached_property
    def NDPI_TAGS(self) -> TiffTagRegistry:
        """Registry of private TIFF tags for Hamamatsu NDPI (65420-65458)."""
        return TiffTagRegistry(((65324, 'OffsetHighBytes'), (65325, 'ByteCountHighBytes'), (65420, 'FileFormat'), (65421, 'Magnification'), (65422, 'XOffsetFromSlideCenter'), (65423, 'YOffsetFromSlideCenter'), (65424, 'ZOffsetFromSlideCenter'), (65425, 'TissueIndex'), (65426, 'McuStarts'), (65427, 'SlideLabel'), (65428, 'AuthCode'), (65429, '65429'), (65430, '65430'), (65431, '65431'), (65432, 'McuStartsHighBytes'), (65433, '65433'), (65434, 'Fluorescence'), (65435, 'ExposureRatio'), (65436, 'RedMultiplier'), (65437, 'GreenMultiplier'), (65438, 'BlueMultiplier'), (65439, 'FocusPoints'), (65440, 'FocusPointRegions'), (65441, 'CaptureMode'), (65442, 'ScannerSerialNumber'), (65443, '65443'), (65444, 'JpegQuality'), (65445, 'RefocusInterval'), (65446, 'FocusOffset'), (65447, 'BlankLines'), (65448, 'FirmwareVersion'), (65449, 'Comments'), (65450, 'LabelObscured'), (65451, 'Wavelength'), (65452, '65452'), (65453, 'LampAge'), (65454, 'ExposureTime'), (65455, 'FocusTime'), (65456, 'ScanTime'), (65457, 'WriteTime'), (65458, 'FullyAutoFocus'), (65500, 'DefaultGamma')))

    @cached_property
    def GPS_TAGS(self) -> TiffTagRegistry:
        """Registry of GPS IFD tags."""
        return TiffTagRegistry(((0, 'GPSVersionID'), (1, 'GPSLatitudeRef'), (2, 'GPSLatitude'), (3, 'GPSLongitudeRef'), (4, 'GPSLongitude'), (5, 'GPSAltitudeRef'), (6, 'GPSAltitude'), (7, 'GPSTimeStamp'), (8, 'GPSSatellites'), (9, 'GPSStatus'), (10, 'GPSMeasureMode'), (11, 'GPSDOP'), (12, 'GPSSpeedRef'), (13, 'GPSSpeed'), (14, 'GPSTrackRef'), (15, 'GPSTrack'), (16, 'GPSImgDirectionRef'), (17, 'GPSImgDirection'), (18, 'GPSMapDatum'), (19, 'GPSDestLatitudeRef'), (20, 'GPSDestLatitude'), (21, 'GPSDestLongitudeRef'), (22, 'GPSDestLongitude'), (23, 'GPSDestBearingRef'), (24, 'GPSDestBearing'), (25, 'GPSDestDistanceRef'), (26, 'GPSDestDistance'), (27, 'GPSProcessingMethod'), (28, 'GPSAreaInformation'), (29, 'GPSDateStamp'), (30, 'GPSDifferential'), (31, 'GPSHPositioningError')))

    @cached_property
    def IOP_TAGS(self) -> TiffTagRegistry:
        """Registry of Interoperability IFD tags."""
        return TiffTagRegistry(((1, 'InteroperabilityIndex'), (2, 'InteroperabilityVersion'), (4096, 'RelatedImageFileFormat'), (4097, 'RelatedImageWidth'), (4098, 'RelatedImageLength')))

    @cached_property
    def PHOTOMETRIC_SAMPLES(self) -> dict[int, int]:
        """Map :py:class:`PHOTOMETRIC` to number of photometric samples."""
        return {0: 1, 1: 1, 2: 3, 3: 1, 4: 1, 5: 4, 6: 3, 8: 3, 9: 3, 10: 3, 32803: 1, 32844: 1, 32845: 3, 34892: 3, 51177: 1, 52527: 1}

    @cached_property
    def DATA_FORMATS(self) -> dict[int, str]:
        """Map :py:class:`DATATYPE` to Python struct formats."""
        return {1: '1B', 2: '1s', 3: '1H', 4: '1I', 5: '2I', 6: '1b', 7: '1B', 8: '1h', 9: '1i', 10: '2i', 11: '1f', 12: '1d', 13: '1I', 16: '1Q', 17: '1q', 18: '1Q'}

    @cached_property
    def DATA_DTYPES(self) -> dict[str, int]:
        """Map NumPy dtype to :py:class:`DATATYPE`."""
        return {'B': 1, 's': 2, 'H': 3, 'I': 4, '2I': 5, 'b': 6, 'h': 8, 'i': 9, '2i': 10, 'f': 11, 'd': 12, 'Q': 16, 'q': 17}

    @cached_property
    def SAMPLE_DTYPES(self) -> dict[tuple[int, int | tuple[int, ...]], str]:
        """Map :py:class:`SAMPLEFORMAT` and BitsPerSample to NumPy dtype."""
        return {(1, 1): '?', (1, 2): 'B', (1, 3): 'B', (1, 4): 'B', (1, 5): 'B', (1, 6): 'B', (1, 7): 'B', (1, 8): 'B', (1, 9): 'H', (1, 10): 'H', (1, 11): 'H', (1, 12): 'H', (1, 13): 'H', (1, 14): 'H', (1, 15): 'H', (1, 16): 'H', (1, 17): 'I', (1, 18): 'I', (1, 19): 'I', (1, 20): 'I', (1, 21): 'I', (1, 22): 'I', (1, 23): 'I', (1, 24): 'I', (1, 25): 'I', (1, 26): 'I', (1, 27): 'I', (1, 28): 'I', (1, 29): 'I', (1, 30): 'I', (1, 31): 'I', (1, 32): 'I', (1, 64): 'Q', (4, 1): '?', (4, 2): 'B', (4, 3): 'B', (4, 4): 'B', (4, 5): 'B', (4, 6): 'B', (4, 7): 'B', (4, 8): 'B', (4, 9): 'H', (4, 10): 'H', (4, 11): 'H', (4, 12): 'H', (4, 13): 'H', (4, 14): 'H', (4, 15): 'H', (4, 16): 'H', (4, 17): 'I', (4, 18): 'I', (4, 19): 'I', (4, 20): 'I', (4, 21): 'I', (4, 22): 'I', (4, 23): 'I', (4, 24): 'I', (4, 25): 'I', (4, 26): 'I', (4, 27): 'I', (4, 28): 'I', (4, 29): 'I', (4, 30): 'I', (4, 31): 'I', (4, 32): 'I', (4, 64): 'Q', (2, 8): 'b', (2, 16): 'h', (2, 32): 'i', (2, 64): 'q', (3, 16): 'e', (3, 24): 'f', (3, 32): 'f', (3, 64): 'd', (6, 64): 'F', (6, 128): 'D', (1, (5, 6, 5)): 'B', (5, 16): 'E', (5, 32): 'F', (5, 64): 'D'}

    @cached_property
    def PREDICTORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`PREDICTOR` value to encode function."""
        return PredictorCodec(True)

    @cached_property
    def UNPREDICTORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`PREDICTOR` value to decode function."""
        return PredictorCodec(False)

    @cached_property
    def COMPRESSORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`COMPRESSION` value to compress function."""
        return CompressionCodec(True)

    @cached_property
    def DECOMPRESSORS(self) -> Mapping[int, Callable[..., Any]]:
        """Map :py:class:`COMPRESSION` value to decompress function."""
        return CompressionCodec(False)

    @cached_property
    def IMAGE_COMPRESSIONS(self) -> set[int]:
        return {6, 7, 22610, 33003, 33004, 33005, 33007, 34712, 34892, 34933, 34934, 48124, 50001, 50002}

    @cached_property
    def AXES_NAMES(self) -> dict[str, str]:
        """Map axes character codes to dimension names.

        - **X : width** (image width)
        - **Y : height** (image length)
        - **Z : depth** (image depth)
        - **S : sample** (color space and extra samples)
        - **I : sequence** (generic sequence of images, frames, planes, pages)
        - **T : time** (time series)
        - **C : channel** (acquisition path or emission wavelength)
        - **A : angle** (OME)
        - **P : phase** (OME. In LSM, **P** maps to **position**)
        - **R : tile** (OME. Region, position, or mosaic)
        - **H : lifetime** (OME. Histogram)
        - **E : lambda** (OME. Excitation wavelength)
        - **Q : other** (OME)
        - **L : exposure** (FluoView)
        - **V : event** (FluoView)
        - **M : mosaic** (LSM 6)
        - **J : column** (NDTiff)
        - **K : row** (NDTiff)

        There is no universal standard for dimension codes or names.
        This mapping mainly follows TIFF, OME-TIFF, ImageJ, LSM, and FluoView
        conventions.

        """
        return {'X': 'width', 'Y': 'height', 'Z': 'depth', 'S': 'sample', 'I': 'sequence', 'T': 'time', 'C': 'channel', 'A': 'angle', 'P': 'phase', 'R': 'tile', 'H': 'lifetime', 'E': 'lambda', 'L': 'exposure', 'V': 'event', 'M': 'mosaic', 'Q': 'other', 'J': 'column', 'K': 'row'}

    @cached_property
    def AXES_CODES(self) -> dict[str, str]:
        """Map dimension names to axes character codes.

        Reverse mapping of :py:attr:`AXES_NAMES`.

        """
        codes = {name: code for code, name in TIFF.AXES_NAMES.items()}
        codes['z'] = 'Z'
        codes['position'] = 'R'
        return codes

    @cached_property
    def AXES_LABELS(self) -> dict[str, str]:
        warnings.warn('<tifffile.TIFF.AXES_LABELS> is deprecated. Use TIFF.AXES_NAMES or TIFF.AXES_CODES.', DeprecationWarning, stacklevel=2)
        return {**TIFF.AXES_NAMES, **TIFF.AXES_CODES}

    @cached_property
    def GEO_KEYS(self) -> type[enum.IntEnum]:
        """:py:class:`geodb.GeoKeys`."""
        try:
            from .geodb import GeoKeys
        except ImportError:

            class GeoKeys(enum.IntEnum):
                pass
        return GeoKeys

    @cached_property
    def GEO_CODES(self) -> dict[int, type[enum.IntEnum]]:
        """Map :py:class:`geodb.GeoKeys` to GeoTIFF codes."""
        try:
            from .geodb import GEO_CODES
        except ImportError:
            GEO_CODES = {}
        return GEO_CODES

    @cached_property
    def PAGE_FLAGS(self) -> set[str]:
        exclude = {'reduced', 'mask', 'final', 'memmappable', 'contiguous', 'tiled', 'subsampled', 'jfif'}
        return {a[3:] for a in dir(TiffPage) if a[:3] == 'is_' and a[3:] not in exclude}

    @cached_property
    def FILE_FLAGS(self) -> set[str]:
        exclude = {'bigtiff', 'appendable'}
        return {a[3:] for a in dir(TiffFile) if a[:3] == 'is_' and a[3:] not in exclude}.union(TIFF.PAGE_FLAGS)

    @property
    def FILE_PATTERNS(self) -> dict[str, str]:
        return {'axes': '(?ix)\n                # matches Olympus OIF and Leica TIFF series\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\\d{1,4}))?\n                '}

    @property
    def FILE_EXTENSIONS(self) -> tuple[str, ...]:
        """Known TIFF file extensions."""
        return ('tif', 'tiff', 'ome.tif', 'lsm', 'stk', 'qpi', 'pcoraw', 'qptiff', 'ptiff', 'ptif', 'gel', 'seq', 'svs', 'scn', 'zif', 'ndpi', 'bif', 'tf8', 'tf2', 'btf', 'eer')

    @property
    def FILEOPEN_FILTER(self) -> list[tuple[str, str]]:
        return [(f'{ext.upper()} files', f'*.{ext}') for ext in TIFF.FILE_EXTENSIONS] + [('allfiles', '*')]

    @property
    def CZ_LSMINFO(self) -> list[tuple[str, str]]:
        return [('MagicNumber', 'u4'), ('StructureSize', 'i4'), ('DimensionX', 'i4'), ('DimensionY', 'i4'), ('DimensionZ', 'i4'), ('DimensionChannels', 'i4'), ('DimensionTime', 'i4'), ('DataType', 'i4'), ('ThumbnailX', 'i4'), ('ThumbnailY', 'i4'), ('VoxelSizeX', 'f8'), ('VoxelSizeY', 'f8'), ('VoxelSizeZ', 'f8'), ('OriginX', 'f8'), ('OriginY', 'f8'), ('OriginZ', 'f8'), ('ScanType', 'u2'), ('SpectralScan', 'u2'), ('TypeOfData', 'u4'), ('OffsetVectorOverlay', 'u4'), ('OffsetInputLut', 'u4'), ('OffsetOutputLut', 'u4'), ('OffsetChannelColors', 'u4'), ('TimeIntervall', 'f8'), ('OffsetChannelDataTypes', 'u4'), ('OffsetScanInformation', 'u4'), ('OffsetKsData', 'u4'), ('OffsetTimeStamps', 'u4'), ('OffsetEventList', 'u4'), ('OffsetRoi', 'u4'), ('OffsetBleachRoi', 'u4'), ('OffsetNextRecording', 'u4'), ('DisplayAspectX', 'f8'), ('DisplayAspectY', 'f8'), ('DisplayAspectZ', 'f8'), ('DisplayAspectTime', 'f8'), ('OffsetMeanOfRoisOverlay', 'u4'), ('OffsetTopoIsolineOverlay', 'u4'), ('OffsetTopoProfileOverlay', 'u4'), ('OffsetLinescanOverlay', 'u4'), ('ToolbarFlags', 'u4'), ('OffsetChannelWavelength', 'u4'), ('OffsetChannelFactors', 'u4'), ('ObjectiveSphereCorrection', 'f8'), ('OffsetUnmixParameters', 'u4'), ('OffsetAcquisitionParameters', 'u4'), ('OffsetCharacteristics', 'u4'), ('OffsetPalette', 'u4'), ('TimeDifferenceX', 'f8'), ('TimeDifferenceY', 'f8'), ('TimeDifferenceZ', 'f8'), ('InternalUse1', 'u4'), ('DimensionP', 'i4'), ('DimensionM', 'i4'), ('DimensionsReserved', '16i4'), ('OffsetTilePositions', 'u4'), ('', '9u4'), ('OffsetPositions', 'u4')]

    @property
    def CZ_LSMINFO_READERS(self) -> dict[str, Callable[[FileHandle], Any] | None]:
        return {'ScanInformation': read_lsm_scaninfo, 'TimeStamps': read_lsm_timestamps, 'EventList': read_lsm_eventlist, 'ChannelColors': read_lsm_channelcolors, 'Positions': read_lsm_positions, 'TilePositions': read_lsm_positions, 'VectorOverlay': None, 'InputLut': read_lsm_lookuptable, 'OutputLut': read_lsm_lookuptable, 'TimeIntervall': None, 'ChannelDataTypes': read_lsm_channeldatatypes, 'KsData': None, 'Roi': None, 'BleachRoi': None, 'NextRecording': None, 'MeanOfRoisOverlay': None, 'TopoIsolineOverlay': None, 'TopoProfileOverlay': None, 'ChannelWavelength': read_lsm_channelwavelength, 'SphereCorrection': None, 'ChannelFactors': None, 'UnmixParameters': None, 'AcquisitionParameters': None, 'Characteristics': None}

    @property
    def CZ_LSMINFO_SCANTYPE(self) -> dict[int, str]:
        return {0: 'XYZCT', 1: 'XYZCT', 2: 'XYZCT', 3: 'XYTCZ', 4: 'XYZTC', 5: 'XYTCZ', 6: 'XYZTC', 7: 'XYCTZ', 8: 'XYCZT', 9: 'XYTCZ', 10: 'XYZCT'}

    @property
    def CZ_LSMINFO_DIMENSIONS(self) -> dict[str, str]:
        return {'X': 'DimensionX', 'Y': 'DimensionY', 'Z': 'DimensionZ', 'C': 'DimensionChannels', 'T': 'DimensionTime', 'P': 'DimensionP', 'M': 'DimensionM'}

    @property
    def CZ_LSMINFO_DATATYPES(self) -> dict[int, str]:
        return {0: 'varying data types', 1: '8 bit unsigned integer', 2: '12 bit unsigned integer', 5: '32 bit float'}

    @property
    def CZ_LSMINFO_TYPEOFDATA(self) -> dict[int, str]:
        return {0: 'Original scan data', 1: 'Calculated data', 2: '3D reconstruction', 3: 'Topography height map'}

    @property
    def CZ_LSMINFO_SCANINFO_ARRAYS(self) -> dict[int, str]:
        return {536870912: 'Tracks', 805306368: 'Lasers', 1610612736: 'DetectionChannels', 2147483648: 'IlluminationChannels', 2684354560: 'BeamSplitters', 3221225472: 'DataChannels', 285212672: 'Timers', 318767104: 'Markers'}

    @property
    def CZ_LSMINFO_SCANINFO_STRUCTS(self) -> dict[int, str]:
        return {1073741824: 'Track', 1342177280: 'Laser', 1879048192: 'DetectionChannel', 2415919104: 'IlluminationChannel', 2952790016: 'BeamSplitter', 3489660928: 'DataChannel', 301989888: 'Timer', 335544320: 'Marker'}

    @property
    def CZ_LSMINFO_SCANINFO_ATTRIBUTES(self) -> dict[int, str]:
        return {268435457: 'Name', 268435458: 'Description', 268435459: 'Notes', 268435460: 'Objective', 268435461: 'ProcessingSummary', 268435462: 'SpecialScanMode', 268435463: 'ScanType', 268435464: 'ScanMode', 268435465: 'NumberOfStacks', 268435466: 'LinesPerPlane', 268435467: 'SamplesPerLine', 268435468: 'PlanesPerVolume', 268435469: 'ImagesWidth', 268435470: 'ImagesHeight', 268435471: 'ImagesNumberPlanes', 268435472: 'ImagesNumberStacks', 268435473: 'ImagesNumberChannels', 268435474: 'LinscanXySize', 268435475: 'ScanDirection', 268435476: 'TimeSeries', 268435477: 'OriginalScanData', 268435478: 'ZoomX', 268435479: 'ZoomY', 268435480: 'ZoomZ', 268435481: 'Sample0X', 268435482: 'Sample0Y', 268435483: 'Sample0Z', 268435484: 'SampleSpacing', 268435485: 'LineSpacing', 268435486: 'PlaneSpacing', 268435487: 'PlaneWidth', 268435488: 'PlaneHeight', 268435489: 'VolumeDepth', 268435491: 'Nutation', 268435508: 'Rotation', 268435509: 'Precession', 268435510: 'Sample0time', 268435511: 'StartScanTriggerIn', 268435512: 'StartScanTriggerOut', 268435513: 'StartScanEvent', 268435520: 'StartScanTime', 268435521: 'StopScanTriggerIn', 268435522: 'StopScanTriggerOut', 268435523: 'StopScanEvent', 268435524: 'StopScanTime', 268435525: 'UseRois', 268435526: 'UseReducedMemoryRois', 268435527: 'User', 268435528: 'UseBcCorrection', 268435529: 'PositionBcCorrection1', 268435536: 'PositionBcCorrection2', 268435537: 'InterpolationY', 268435538: 'CameraBinning', 268435539: 'CameraSupersampling', 268435540: 'CameraFrameWidth', 268435541: 'CameraFrameHeight', 268435542: 'CameraOffsetX', 268435543: 'CameraOffsetY', 268435545: 'RtBinning', 268435546: 'RtFrameWidth', 268435547: 'RtFrameHeight', 268435548: 'RtRegionWidth', 268435549: 'RtRegionHeight', 268435550: 'RtOffsetX', 268435551: 'RtOffsetY', 268435552: 'RtZoom', 268435553: 'RtLinePeriod', 268435554: 'Prescan', 268435555: 'ScanDirectionZ', 1073741825: 'MultiplexType', 1073741826: 'MultiplexOrder', 1073741827: 'SamplingMode', 1073741828: 'SamplingMethod', 1073741829: 'SamplingNumber', 1073741830: 'Acquire', 1073741831: 'SampleObservationTime', 1073741835: 'TimeBetweenStacks', 1073741836: 'Name', 1073741837: 'Collimator1Name', 1073741838: 'Collimator1Position', 1073741839: 'Collimator2Name', 1073741840: 'Collimator2Position', 1073741841: 'IsBleachTrack', 1073741842: 'IsBleachAfterScanNumber', 1073741843: 'BleachScanNumber', 1073741844: 'TriggerIn', 1073741845: 'TriggerOut', 1073741846: 'IsRatioTrack', 1073741847: 'BleachCount', 1073741848: 'SpiCenterWavelength', 1073741849: 'PixelTime', 1073741857: 'CondensorFrontlens', 1073741859: 'FieldStopValue', 1073741860: 'IdCondensorAperture', 1073741861: 'CondensorAperture', 1073741862: 'IdCondensorRevolver', 1073741863: 'CondensorFilter', 1073741864: 'IdTransmissionFilter1', 1073741865: 'IdTransmission1', 1073741872: 'IdTransmissionFilter2', 1073741873: 'IdTransmission2', 1073741874: 'RepeatBleach', 1073741875: 'EnableSpotBleachPos', 1073741876: 'SpotBleachPosx', 1073741877: 'SpotBleachPosy', 1073741878: 'SpotBleachPosz', 1073741879: 'IdTubelens', 1073741880: 'IdTubelensPosition', 1073741881: 'TransmittedLight', 1073741882: 'ReflectedLight', 1073741883: 'SimultanGrabAndBleach', 1073741884: 'BleachPixelTime', 1342177281: 'Name', 1342177282: 'Acquire', 1342177283: 'Power', 1879048193: 'IntegrationMode', 1879048194: 'SpecialMode', 1879048195: 'DetectorGainFirst', 1879048196: 'DetectorGainLast', 1879048197: 'AmplifierGainFirst', 1879048198: 'AmplifierGainLast', 1879048199: 'AmplifierOffsFirst', 1879048200: 'AmplifierOffsLast', 1879048201: 'PinholeDiameter', 1879048202: 'CountingTrigger', 1879048203: 'Acquire', 1879048204: 'PointDetectorName', 1879048205: 'AmplifierName', 1879048206: 'PinholeName', 1879048207: 'FilterSetName', 1879048208: 'FilterName', 1879048211: 'IntegratorName', 1879048212: 'ChannelName', 1879048213: 'DetectorGainBc1', 1879048214: 'DetectorGainBc2', 1879048215: 'AmplifierGainBc1', 1879048216: 'AmplifierGainBc2', 1879048217: 'AmplifierOffsetBc1', 1879048224: 'AmplifierOffsetBc2', 1879048225: 'SpectralScanChannels', 1879048226: 'SpiWavelengthStart', 1879048227: 'SpiWavelengthStop', 1879048230: 'DyeName', 1879048231: 'DyeFolder', 2415919105: 'Name', 2415919106: 'Power', 2415919107: 'Wavelength', 2415919108: 'Aquire', 2415919109: 'DetchannelName', 2415919110: 'PowerBc1', 2415919111: 'PowerBc2', 2952790017: 'FilterSet', 2952790018: 'Filter', 2952790019: 'Name', 3489660929: 'Name', 3489660931: 'Acquire', 3489660932: 'Color', 3489660933: 'SampleType', 3489660934: 'BitsPerSample', 3489660935: 'RatioType', 3489660936: 'RatioTrack1', 3489660937: 'RatioTrack2', 3489660938: 'RatioChannel1', 3489660939: 'RatioChannel2', 3489660940: 'RatioConst1', 3489660941: 'RatioConst2', 3489660942: 'RatioConst3', 3489660943: 'RatioConst4', 3489660944: 'RatioConst5', 3489660945: 'RatioConst6', 3489660946: 'RatioFirstImages1', 3489660947: 'RatioFirstImages2', 3489660948: 'DyeName', 3489660949: 'DyeFolder', 3489660950: 'Spectrum', 3489660951: 'Acquire', 301989889: 'Name', 301989890: 'Description', 301989891: 'Interval', 301989892: 'TriggerIn', 301989893: 'TriggerOut', 301989894: 'ActivationTime', 301989895: 'ActivationNumber', 335544321: 'Name', 335544322: 'Description', 335544323: 'TriggerIn', 335544324: 'TriggerOut'}

    @cached_property
    def CZ_LSM_LUTTYPE(self):

        class CZ_LSM_LUTTYPE(enum.IntEnum):
            NORMAL = 0
            ORIGINAL = 1
            RAMP = 2
            POLYLINE = 3
            SPLINE = 4
            GAMMA = 5
        return CZ_LSM_LUTTYPE

    @cached_property
    def CZ_LSM_SUBBLOCK_TYPE(self):

        class CZ_LSM_SUBBLOCK_TYPE(enum.IntEnum):
            END = 0
            GAMMA = 1
            BRIGHTNESS = 2
            CONTRAST = 3
            RAMP = 4
            KNOTS = 5
            PALETTE_12_TO_12 = 6
        return CZ_LSM_SUBBLOCK_TYPE

    @property
    def NIH_IMAGE_HEADER(self):
        return [('FileID', 'a8'), ('nLines', 'i2'), ('PixelsPerLine', 'i2'), ('Version', 'i2'), ('OldLutMode', 'i2'), ('OldnColors', 'i2'), ('Colors', 'u1', (3, 32)), ('OldColorStart', 'i2'), ('ColorWidth', 'i2'), ('ExtraColors', 'u2', (6, 3)), ('nExtraColors', 'i2'), ('ForegroundIndex', 'i2'), ('BackgroundIndex', 'i2'), ('XScale', 'f8'), ('Unused2', 'i2'), ('Unused3', 'i2'), ('UnitsID', 'i2'), ('p1', [('x', 'i2'), ('y', 'i2')]), ('p2', [('x', 'i2'), ('y', 'i2')]), ('CurveFitType', 'i2'), ('nCoefficients', 'i2'), ('Coeff', 'f8', 6), ('UMsize', 'u1'), ('UM', 'a15'), ('UnusedBoolean', 'u1'), ('BinaryPic', 'b1'), ('SliceStart', 'i2'), ('SliceEnd', 'i2'), ('ScaleMagnification', 'f4'), ('nSlices', 'i2'), ('SliceSpacing', 'f4'), ('CurrentSlice', 'i2'), ('FrameInterval', 'f4'), ('PixelAspectRatio', 'f4'), ('ColorStart', 'i2'), ('ColorEnd', 'i2'), ('nColors', 'i2'), ('Fill1', '3u2'), ('Fill2', '3u2'), ('Table', 'u1'), ('LutMode', 'u1'), ('InvertedTable', 'b1'), ('ZeroClip', 'b1'), ('XUnitSize', 'u1'), ('XUnit', 'a11'), ('StackType', 'i2')]

    @property
    def NIH_COLORTABLE_TYPE(self) -> tuple[str, ...]:
        return ('CustomTable', 'AppleDefault', 'Pseudo20', 'Pseudo32', 'Rainbow', 'Fire1', 'Fire2', 'Ice', 'Grays', 'Spectrum')

    @property
    def NIH_LUTMODE_TYPE(self) -> tuple[str, ...]:
        return ('PseudoColor', 'OldAppleDefault', 'OldSpectrum', 'GrayScale', 'ColorLut', 'CustomGrayscale')

    @property
    def NIH_CURVEFIT_TYPE(self) -> tuple[str, ...]:
        return ('StraightLine', 'Poly2', 'Poly3', 'Poly4', 'Poly5', 'ExpoFit', 'PowerFit', 'LogFit', 'RodbardFit', 'SpareFit1', 'Uncalibrated', 'UncalibratedOD')

    @property
    def NIH_UNITS_TYPE(self) -> tuple[str, ...]:
        return ('Nanometers', 'Micrometers', 'Millimeters', 'Centimeters', 'Meters', 'Kilometers', 'Inches', 'Feet', 'Miles', 'Pixels', 'OtherUnits')

    @property
    def TVIPS_HEADER_V1(self) -> list[tuple[str, str]]:
        return [('Version', 'i4'), ('CommentV1', 'a80'), ('HighTension', 'i4'), ('SphericalAberration', 'i4'), ('IlluminationAperture', 'i4'), ('Magnification', 'i4'), ('PostMagnification', 'i4'), ('FocalLength', 'i4'), ('Defocus', 'i4'), ('Astigmatism', 'i4'), ('AstigmatismDirection', 'i4'), ('BiprismVoltage', 'i4'), ('SpecimenTiltAngle', 'i4'), ('SpecimenTiltDirection', 'i4'), ('IlluminationTiltDirection', 'i4'), ('IlluminationTiltAngle', 'i4'), ('ImageMode', 'i4'), ('EnergySpread', 'i4'), ('ChromaticAberration', 'i4'), ('ShutterType', 'i4'), ('DefocusSpread', 'i4'), ('CcdNumber', 'i4'), ('CcdSize', 'i4'), ('OffsetXV1', 'i4'), ('OffsetYV1', 'i4'), ('PhysicalPixelSize', 'i4'), ('Binning', 'i4'), ('ReadoutSpeed', 'i4'), ('GainV1', 'i4'), ('SensitivityV1', 'i4'), ('ExposureTimeV1', 'i4'), ('FlatCorrected', 'i4'), ('DeadPxCorrected', 'i4'), ('ImageMean', 'i4'), ('ImageStd', 'i4'), ('DisplacementX', 'i4'), ('DisplacementY', 'i4'), ('DateV1', 'i4'), ('TimeV1', 'i4'), ('ImageMin', 'i4'), ('ImageMax', 'i4'), ('ImageStatisticsQuality', 'i4')]

    @property
    def TVIPS_HEADER_V2(self) -> list[tuple[str, str]]:
        return [('ImageName', 'V160'), ('ImageFolder', 'V160'), ('ImageSizeX', 'i4'), ('ImageSizeY', 'i4'), ('ImageSizeZ', 'i4'), ('ImageSizeE', 'i4'), ('ImageDataType', 'i4'), ('Date', 'i4'), ('Time', 'i4'), ('Comment', 'V1024'), ('ImageHistory', 'V1024'), ('Scaling', '16f4'), ('ImageStatistics', '16c16'), ('ImageType', 'i4'), ('ImageDisplaType', 'i4'), ('PixelSizeX', 'f4'), ('PixelSizeY', 'f4'), ('ImageDistanceZ', 'f4'), ('ImageDistanceE', 'f4'), ('ImageMisc', '32f4'), ('TemType', 'V160'), ('TemHighTension', 'f4'), ('TemAberrations', '32f4'), ('TemEnergy', '32f4'), ('TemMode', 'i4'), ('TemMagnification', 'f4'), ('TemMagnificationCorrection', 'f4'), ('PostMagnification', 'f4'), ('TemStageType', 'i4'), ('TemStagePosition', '5f4'), ('TemImageShift', '2f4'), ('TemBeamShift', '2f4'), ('TemBeamTilt', '2f4'), ('TilingParameters', '7f4'), ('TemIllumination', '3f4'), ('TemShutter', 'i4'), ('TemMisc', '32f4'), ('CameraType', 'V160'), ('PhysicalPixelSizeX', 'f4'), ('PhysicalPixelSizeY', 'f4'), ('OffsetX', 'i4'), ('OffsetY', 'i4'), ('BinningX', 'i4'), ('BinningY', 'i4'), ('ExposureTime', 'f4'), ('Gain', 'f4'), ('ReadoutRate', 'f4'), ('FlatfieldDescription', 'V160'), ('Sensitivity', 'f4'), ('Dose', 'f4'), ('CamMisc', '32f4'), ('FeiMicroscopeInformation', 'V1024'), ('FeiSpecimenInformation', 'V1024'), ('Magic', 'u4')]

    @property
    def MM_HEADER(self) -> list[tuple[Any, ...]]:
        MM_DIMENSION = [('Name', 'a16'), ('Size', 'i4'), ('Origin', 'f8'), ('Resolution', 'f8'), ('Unit', 'a64')]
        return [('HeaderFlag', 'i2'), ('ImageType', 'u1'), ('ImageName', 'a257'), ('OffsetData', 'u4'), ('PaletteSize', 'i4'), ('OffsetPalette0', 'u4'), ('OffsetPalette1', 'u4'), ('CommentSize', 'i4'), ('OffsetComment', 'u4'), ('Dimensions', MM_DIMENSION, 10), ('OffsetPosition', 'u4'), ('MapType', 'i2'), ('MapMin', 'f8'), ('MapMax', 'f8'), ('MinValue', 'f8'), ('MaxValue', 'f8'), ('OffsetMap', 'u4'), ('Gamma', 'f8'), ('Offset', 'f8'), ('GrayChannel', MM_DIMENSION), ('OffsetThumbnail', 'u4'), ('VoiceField', 'i4'), ('OffsetVoiceField', 'u4')]

    @property
    def MM_DIMENSIONS(self) -> dict[str, str]:
        return {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'T': 'T', 'CH': 'C', 'WAVELENGTH': 'C', 'TIME': 'T', 'XY': 'R', 'EVENT': 'V', 'EXPOSURE': 'L'}

    @property
    def UIC_TAGS(self) -> list[tuple[str, Any]]:
        from fractions import Fraction
        return [('AutoScale', int), ('MinScale', int), ('MaxScale', int), ('SpatialCalibration', int), ('XCalibration', Fraction), ('YCalibration', Fraction), ('CalibrationUnits', str), ('Name', str), ('ThreshState', int), ('ThreshStateRed', int), ('tagid_10', None), ('ThreshStateGreen', int), ('ThreshStateBlue', int), ('ThreshStateLo', int), ('ThreshStateHi', int), ('Zoom', int), ('CreateTime', julian_datetime), ('LastSavedTime', julian_datetime), ('currentBuffer', int), ('grayFit', None), ('grayPointCount', None), ('grayX', Fraction), ('grayY', Fraction), ('grayMin', Fraction), ('grayMax', Fraction), ('grayUnitName', str), ('StandardLUT', int), ('wavelength', int), ('StagePosition', '(%i,2,2)u4'), ('CameraChipOffset', '(%i,2,2)u4'), ('OverlayMask', None), ('OverlayCompress', None), ('Overlay', None), ('SpecialOverlayMask', None), ('SpecialOverlayCompress', None), ('SpecialOverlay', None), ('ImageProperty', read_uic_image_property), ('StageLabel', '%ip'), ('AutoScaleLoInfo', Fraction), ('AutoScaleHiInfo', Fraction), ('AbsoluteZ', '(%i,2)u4'), ('AbsoluteZValid', '(%i,)u4'), ('Gamma', 'I'), ('GammaRed', 'I'), ('GammaGreen', 'I'), ('GammaBlue', 'I'), ('CameraBin', '2I'), ('NewLUT', int), ('ImagePropertyEx', None), ('PlaneProperty', int), ('UserLutTable', '(256,3)u1'), ('RedAutoScaleInfo', int), ('RedAutoScaleLoInfo', Fraction), ('RedAutoScaleHiInfo', Fraction), ('RedMinScaleInfo', int), ('RedMaxScaleInfo', int), ('GreenAutoScaleInfo', int), ('GreenAutoScaleLoInfo', Fraction), ('GreenAutoScaleHiInfo', Fraction), ('GreenMinScaleInfo', int), ('GreenMaxScaleInfo', int), ('BlueAutoScaleInfo', int), ('BlueAutoScaleLoInfo', Fraction), ('BlueAutoScaleHiInfo', Fraction), ('BlueMinScaleInfo', int), ('BlueMaxScaleInfo', int)]

    @property
    def PILATUS_HEADER(self) -> dict[str, Any]:
        return {'Detector': ([slice(1, None)], str), 'Pixel_size': ([1, 4], float), 'Silicon': ([3], float), 'Exposure_time': ([1], float), 'Exposure_period': ([1], float), 'Tau': ([1], float), 'Count_cutoff': ([1], int), 'Threshold_setting': ([1], float), 'Gain_setting': ([1, 2], str), 'N_excluded_pixels': ([1], int), 'Excluded_pixels': ([1], str), 'Flat_field': ([1], str), 'Trim_file': ([1], str), 'Image_path': ([1], str), 'Wavelength': ([1], float), 'Energy_range': ([1, 2], float), 'Detector_distance': ([1], float), 'Detector_Voffset': ([1], float), 'Beam_xy': ([1, 2], float), 'Flux': ([1], str), 'Filter_transmission': ([1], float), 'Start_angle': ([1], float), 'Angle_increment': ([1], float), 'Detector_2theta': ([1], float), 'Polarization': ([1], float), 'Alpha': ([1], float), 'Kappa': ([1], float), 'Phi': ([1], float), 'Phi_increment': ([1], float), 'Chi': ([1], float), 'Chi_increment': ([1], float), 'Oscillation_axis': ([slice(1, None)], str), 'N_oscillations': ([1], int), 'Start_position': ([1], float), 'Position_increment': ([1], float), 'Shutter_time': ([1], float), 'Omega': ([1], float), 'Omega_increment': ([1], float)}

    @cached_property
    def ALLOCATIONGRANULARITY(self) -> int:
        import mmap
        return mmap.ALLOCATIONGRANULARITY

    @cached_property
    def MAXWORKERS(self) -> int:
        """Default maximum number of threads for de/compressing segments.

        The value of the ``TIFFFILE_NUM_THREADS`` environment variable if set,
        else half the CPU cores up to 32.

        """
        if 'TIFFFILE_NUM_THREADS' in os.environ:
            return max(1, int(os.environ['TIFFFILE_NUM_THREADS']))
        cpu_count: int | None
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 1
        return min(32, max(1, cpu_count // 2))

    @cached_property
    def MAXIOWORKERS(self) -> int:
        """Default maximum number of I/O threads for reading file sequences.

        The value of the ``TIFFFILE_NUM_IOTHREADS`` environment variable if
        set, else 4 more than the number of CPU cores up to 32.

        """
        if 'TIFFFILE_NUM_IOTHREADS' in os.environ:
            return max(1, int(os.environ['TIFFFILE_NUM_IOTHREADS']))
        cpu_count: int | None
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 5
        return min(32, cpu_count + 4)
    BUFFERSIZE: int = 268435456
    'Default number of bytes to read or encode in one pass (256 MB).'

    @cached_property
    def CHUNKMODE(self) -> type[CHUNKMODE]:
        """Deprecated alias of :py:class:`CHUNKMODE`."""
        return CHUNKMODE

    @cached_property
    def COMPRESSION(self) -> type[COMPRESSION]:
        """Deprecated alias of :py:class:`COMPRESSION`."""
        return COMPRESSION

    @cached_property
    def PREDICTOR(self) -> type[PREDICTOR]:
        """Deprecated alias of :py:class:`PREDICTOR`."""
        return PREDICTOR

    @cached_property
    def EXTRASAMPLE(self) -> type[EXTRASAMPLE]:
        """Deprecated alias of :py:class:`EXTRASAMPLE`."""
        return EXTRASAMPLE

    @cached_property
    def FILETYPE(self) -> type[FILETYPE]:
        """Deprecated alias of :py:class:`FILETYPE`."""
        return FILETYPE

    @cached_property
    def FILLORDER(self) -> type[FILLORDER]:
        """Deprecated alias of :py:class:`FILLORDER`."""
        return FILLORDER

    @cached_property
    def PHOTOMETRIC(self) -> type[PHOTOMETRIC]:
        """Deprecated alias of :py:class:`PHOTOMETRIC`."""
        return PHOTOMETRIC

    @cached_property
    def PLANARCONFIG(self) -> type[PLANARCONFIG]:
        """Deprecated alias of :py:class:`PLANARCONFIG`."""
        return PLANARCONFIG

    @cached_property
    def RESUNIT(self) -> type[RESUNIT]:
        """Deprecated alias of :py:class:`RESUNIT`."""
        return RESUNIT

    @cached_property
    def ORIENTATION(self) -> type[ORIENTATION]:
        """Deprecated alias of :py:class:`ORIENTATION`."""
        return ORIENTATION

    @cached_property
    def SAMPLEFORMAT(self) -> type[SAMPLEFORMAT]:
        """Deprecated alias of :py:class:`SAMPLEFORMAT`."""
        return SAMPLEFORMAT

    @cached_property
    def DATATYPES(self) -> type[DATATYPE]:
        """Deprecated alias of :py:class:`DATATYPE`."""
        return DATATYPE