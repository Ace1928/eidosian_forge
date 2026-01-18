import unittest
import gyp.MSVSSettings as MSVSSettings
from io import StringIO
Tests the conversion of an actual project.

    A VS2008 project with most of the options defined was created through the
    VS2008 IDE.  It was then converted to VS2010.  The tool settings found in
    the .vcproj and .vcxproj files were converted to the two dictionaries
    msvs_settings and expected_msbuild_settings.

    Note that for many settings, the VS2010 converter adds macros like
    %(AdditionalIncludeDirectories) to make sure than inherited values are
    included.  Since the Gyp projects we generate do not use inheritance,
    we removed these macros.  They were:
        ClCompile:
            AdditionalIncludeDirectories:  ';%(AdditionalIncludeDirectories)'
            AdditionalOptions:  ' %(AdditionalOptions)'
            AdditionalUsingDirectories:  ';%(AdditionalUsingDirectories)'
            DisableSpecificWarnings: ';%(DisableSpecificWarnings)',
            ForcedIncludeFiles:  ';%(ForcedIncludeFiles)',
            ForcedUsingFiles:  ';%(ForcedUsingFiles)',
            PreprocessorDefinitions:  ';%(PreprocessorDefinitions)',
            UndefinePreprocessorDefinitions:
                ';%(UndefinePreprocessorDefinitions)',
        Link:
            AdditionalDependencies:  ';%(AdditionalDependencies)',
            AdditionalLibraryDirectories:  ';%(AdditionalLibraryDirectories)',
            AdditionalManifestDependencies:
                ';%(AdditionalManifestDependencies)',
            AdditionalOptions:  ' %(AdditionalOptions)',
            AddModuleNamesToAssembly:  ';%(AddModuleNamesToAssembly)',
            AssemblyLinkResource:  ';%(AssemblyLinkResource)',
            DelayLoadDLLs:  ';%(DelayLoadDLLs)',
            EmbedManagedResourceFile:  ';%(EmbedManagedResourceFile)',
            ForceSymbolReferences:  ';%(ForceSymbolReferences)',
            IgnoreSpecificDefaultLibraries:
                ';%(IgnoreSpecificDefaultLibraries)',
        ResourceCompile:
            AdditionalIncludeDirectories:  ';%(AdditionalIncludeDirectories)',
            AdditionalOptions:  ' %(AdditionalOptions)',
            PreprocessorDefinitions:  ';%(PreprocessorDefinitions)',
        Manifest:
            AdditionalManifestFiles:  ';%(AdditionalManifestFiles)',
            AdditionalOptions:  ' %(AdditionalOptions)',
            InputResourceManifests:  ';%(InputResourceManifests)',
    