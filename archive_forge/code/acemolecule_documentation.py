import os
from copy import deepcopy
from ase.io import read
from ase.calculators.calculator import ReadError
from ase.calculators.calculator import FileIOCalculator
Write ACE-Molecule input

        ACE-Molecule input examples (not minimal)
        %% BasicInformation
            Type    Scaling
            Scaling 0.4
            Basis   Sinc
            Cell    10.0
            Grid    Sphere
            GeometryFormat      xyz
            SpinMultiplicity    3.0
            Polarize    1
            Centered    0
            %% Pseudopotential
                Pseudopotential 1
                UsingDoubleGrid 0
                FilterType      Sinc
                Format          upf
                PSFilePath      /PATH/TO/UPF
                PSFileSuffix    .pbe-theos.UPF
            %% End
            GeometryFilename    xyz/C.xyz
        %% End
        %% Guess
            InitialGuess        3
            InitialFilenames    001.cube
            InitialFilenames    002.cube
        %% End
        %% Scf
            IterateMaxCycle     150
            ConvergenceType     Energy
            ConvergenceTolerance    0.00001
            EnergyDecomposition     1
            ComputeInitialEnergy    1
            %% Diagonalize
                Tolerance           0.000001
            %% End
            %% ExchangeCorrelation
                XFunctional     GGA_X_PBE
                CFunctional     GGA_C_PBE
            %% End
            %% Mixing
                MixingMethod         1
                MixingType           Density
                MixingParameter      0.5
                PulayMixingParameter 0.1
            %% End
        %% End

        Parameters
        ==========
        fpt: File object, should be write mode.
        param: Dictionary of parameters. Also should contain special 'order' section_name for parameter section ordering.
        depth: Nested input depth.

        Notes
        =====
         - Order of parameter section (denoted using %% -- %% BasicInformation, %% Guess, etc.) is important, because it determines calculation order.
           For example, if Guess section comes after Scf section, calculation will not run because Scf will tries to run without initial Hamiltonian.
         - Order of each parameter section-section_name pair is not important unless their keys are the same.
         - Indentation unimportant and capital letters are important.
        