from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.serialization import dumpfn, loadfn
from tqdm import tqdm
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine, Spin
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Orbital
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
class BztTransportProperties:
    """Compute Seebeck, Conductivity, Electrical part of thermal conductivity
    and Hall coefficient, conductivity effective mass, Power Factor tensors
    w.r.t. the chemical potential and temperatures, from dft band structure via
    interpolation.
    """

    def __init__(self, BztInterpolator, temp_r=None, doping=None, npts_mu=4000, CRTA=1e-14, margin=None, save_bztTranspProps=False, load_bztTranspProps=False, fname='bztTranspProps.json.gz') -> None:
        """
        Args:
            BztInterpolator: a BztInterpolator previously generated
            temp_r: numpy array of temperatures at which to calculate transport properties
            doping: doping levels at which to calculate transport properties. If provided,
                transport properties w.r.t. these doping levels are also computed. See
                compute_properties_doping() method for details.
            npts_mu: number of energy points at which to calculate transport properties
            CRTA: constant value of the relaxation time
            margin: The energy range of the interpolation is extended by this value on both sides.
                Defaults to 9 * units.BOLTZMANN * temp_r.max().
            save_bztTranspProps: Default False. If True all computed transport properties
                will be stored in fname file.
            load_bztTranspProps: Default False. If True all computed transport properties
                will be loaded from fname file.
            fname: File path where to save/load transport properties.

        Upon creation, it contains properties tensors w.r.t. the chemical potential
        of size (len(temp_r),npts_mu,3,3):
            Conductivity_mu (S/m), Seebeck_mu (microV/K), Kappa_mu (W/(m*K)),
            Power_Factor_mu (milliW/K m);
            cond_Effective_mass_mu (m_e) calculated as Ref.
        Also:
            Carrier_conc_mu: carrier concentration of size (len(temp_r),npts_mu)
            Hall_carrier_conc_trace_mu: trace of Hall carrier concentration of size
                (len(temp_r),npts_mu)
            mu_r_eV: array of energies in eV and with E_fermi at 0.0
                where all the properties are calculated.

        Example:
            bztTransp = BztTransportProperties(bztInterp,temp_r = np.arange(100,1400,100))
        """
        if temp_r is None:
            temp_r = np.arange(100, 1400, 100)
        self.dosweight = BztInterpolator.data.dosweight
        self.volume = BztInterpolator.data.get_volume()
        self.nelect = BztInterpolator.data.nelect
        self.efermi = BztInterpolator.data.fermi / units.eV
        if margin is None:
            margin = 9 * units.BOLTZMANN * temp_r.max()
        if load_bztTranspProps:
            self.load(fname)
        else:
            self.CRTA = CRTA
            self.temp_r = temp_r
            self.doping = doping
            self.epsilon, self.dos, self.vvdos, self.cdos = BL.BTPDOS(BztInterpolator.eband, BztInterpolator.vvband, npts=npts_mu, cband=BztInterpolator.cband)
            mur_indices = np.logical_and(self.epsilon > self.epsilon.min() + margin, self.epsilon < self.epsilon.max() - margin)
            self.mu_r = self.epsilon[mur_indices]
            self.mu_r_eV = self.mu_r / units.eV - self.efermi
            N, L0, L1, L2, Lm11 = BL.fermiintegrals(self.epsilon, self.dos, self.vvdos, mur=self.mu_r, Tr=temp_r, dosweight=self.dosweight, cdos=self.cdos)
            self.Conductivity_mu, self.Seebeck_mu, self.Kappa_mu, Hall_mu = BL.calc_Onsager_coefficients(L0, L1, L2, self.mu_r, temp_r, self.volume, Lm11=Lm11)
            self.Conductivity_mu *= CRTA
            self.Seebeck_mu *= 1000000.0
            self.Kappa_mu *= CRTA
            self.Hall_carrier_conc_trace_mu = units.Coulomb * 1e-06 / (np.abs(Hall_mu[:, :, 0, 1, 2] + Hall_mu[:, :, 2, 0, 1] + Hall_mu[:, :, 1, 2, 0]) / 3)
            self.Carrier_conc_mu = (N + self.nelect) / (self.volume / (units.Meter / 100.0) ** 3)
            cond_eff_mass = np.zeros((len(self.temp_r), len(self.mu_r), 3, 3))
            for temp in range(len(self.temp_r)):
                for i in range(len(self.mu_r)):
                    try:
                        cond_eff_mass[temp, i] = np.linalg.inv(self.Conductivity_mu[temp, i]) * self.Carrier_conc_mu[temp, i] * units.qe_SI ** 2 / units.me_SI * 1000000.0
                    except np.linalg.LinAlgError:
                        pass
            self.Effective_mass_mu = cond_eff_mass * CRTA
            self.Power_Factor_mu = self.Seebeck_mu @ self.Seebeck_mu @ self.Conductivity_mu
            self.Power_Factor_mu *= 1e-09
            self.contain_props_doping = False
            if isinstance(doping, np.ndarray):
                self.compute_properties_doping(doping, temp_r)
            if save_bztTranspProps:
                self.save(fname)

    def compute_properties_doping(self, doping, temp_r=None) -> None:
        """Calculate all the properties w.r.t. the doping levels in input.

        Args:
            doping: numpy array specifying the doping levels
            temp_r: numpy array specifying the temperatures

        When executed, it add the following variable at the BztTransportProperties
        object:
            Conductivity_doping, Seebeck_doping, Kappa_doping, Power_Factor_doping,
            cond_Effective_mass_doping are dictionaries with 'n' and 'p' keys and
            arrays of dim (len(temp_r),len(doping),3,3) as values.
            Carriers_conc_doping: carriers concentration for each doping level and T.
            mu_doping_eV: the chemical potential correspondent to each doping level.
        """
        if temp_r is None:
            temp_r = self.temp_r
        self.Conductivity_doping, self.Seebeck_doping, self.Kappa_doping, self.Carriers_conc_doping = ({}, {}, {}, {})
        self.Power_Factor_doping, self.Effective_mass_doping = ({}, {})
        mu_doping = {}
        doping_carriers = [dop * (self.volume / (units.Meter / 100.0) ** 3) for dop in doping]
        for dop_type in ['n', 'p']:
            sbk = np.zeros((len(temp_r), len(doping), 3, 3))
            cond = np.zeros((len(temp_r), len(doping), 3, 3))
            kappa = np.zeros((len(temp_r), len(doping), 3, 3))
            hall = np.zeros((len(temp_r), len(doping), 3, 3, 3))
            dc = np.zeros((len(temp_r), len(doping)))
            if dop_type == 'p':
                doping_carriers = [-dop for dop in doping_carriers]
            mu_doping[dop_type] = np.zeros((len(temp_r), len(doping)))
            for idx_t, temp in enumerate(temp_r):
                for idx_d, dop_car in enumerate(doping_carriers):
                    mu_doping[dop_type][idx_t, idx_d] = BL.solve_for_mu(self.epsilon, self.dos, self.nelect + dop_car, temp, self.dosweight, True, False)
                N, L0, L1, L2, Lm11 = BL.fermiintegrals(self.epsilon, self.dos, self.vvdos, mur=mu_doping[dop_type][idx_t], Tr=np.array([temp]), dosweight=self.dosweight)
                cond[idx_t], sbk[idx_t], kappa[idx_t], hall[idx_t] = BL.calc_Onsager_coefficients(L0, L1, L2, mu_doping[dop_type][idx_t], np.array([temp]), self.volume, Lm11)
                dc[idx_t] = self.nelect + N
            self.Conductivity_doping[dop_type] = cond * self.CRTA
            self.Seebeck_doping[dop_type] = sbk * 1000000.0
            self.Kappa_doping[dop_type] = kappa * self.CRTA
            self.Carriers_conc_doping[dop_type] = dc / (self.volume / (units.Meter / 100.0) ** 3)
            self.Power_Factor_doping[dop_type] = sbk @ sbk @ cond * self.CRTA * 1000.0
            cond_eff_mass = np.zeros((len(temp_r), len(doping), 3, 3))
            for idx_t in range(len(temp_r)):
                for idx_d, dop in enumerate(doping):
                    try:
                        cond_eff_mass[idx_t, idx_d] = np.linalg.inv(cond[idx_t, idx_d]) * dop * units.qe_SI ** 2 / units.me_SI * 1000000.0
                    except np.linalg.LinAlgError:
                        pass
            self.Effective_mass_doping[dop_type] = cond_eff_mass
        self.doping = doping
        self.mu_doping = mu_doping
        self.mu_doping_eV = {k: v / units.eV - self.efermi for k, v in mu_doping.items()}
        self.contain_props_doping = True

    def save(self, fname='bztTranspProps.json.gz') -> None:
        """Save the transport properties to fname file."""
        lst_props = [self.temp_r, self.CRTA, self.epsilon, self.dos, self.vvdos, self.cdos, self.mu_r, self.mu_r_eV, self.Conductivity_mu, self.Seebeck_mu, self.Kappa_mu, self.Carrier_conc_mu, self.Hall_carrier_conc_trace_mu, self.Power_Factor_mu, self.Effective_mass_mu]
        if self.contain_props_doping:
            props = [self.Conductivity_doping, self.Seebeck_doping, self.Kappa_doping, self.Power_Factor_doping, self.Effective_mass_doping, self.Carriers_conc_doping, self.doping, self.mu_doping, self.mu_doping_eV]
            lst_props.extend(props)
        dumpfn(lst_props, fname)

    def load(self, fname='bztTranspProps.json.gz') -> bool:
        """Load the transport properties from fname file."""
        lst = loadfn(fname)
        self.temp_r, self.CRTA, self.epsilon, self.dos, self.vvdos, self.cdos, self.mu_r, self.mu_r_eV, self.Conductivity_mu, self.Seebeck_mu, self.Kappa_mu, self.Carrier_conc_mu, self.Hall_carrier_conc_trace_mu, self.Power_Factor_mu, self.Effective_mass_mu = lst[:15]
        if len(lst) > 15:
            self.Conductivity_doping, self.Seebeck_doping, self.Kappa_doping, self.Power_Factor_doping, self.Effective_mass_doping, self.Carriers_conc_doping, self.doping, self.mu_doping, self.mu_doping_eV = lst[15:]
            self.contains_doping_props = True
        return True