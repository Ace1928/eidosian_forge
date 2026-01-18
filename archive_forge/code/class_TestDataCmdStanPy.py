import os
import sys
import tempfile
from glob import glob
import numpy as np
import pytest
from ... import from_cmdstanpy
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.skipif(sys.version_info < (3, 6), reason='CmdStanPy is supported only Python 3.6+')
class TestDataCmdStanPy:

    @pytest.fixture(scope='session')
    def data_directory(self):
        here = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(here, '..', 'saved_models')
        return data_directory

    @pytest.fixture(scope='class')
    def filepaths(self, data_directory):
        files = {'nowarmup': glob(os.path.join(data_directory, 'cmdstanpy', 'cmdstanpy_eight_schools_nowarmup-[1-4].csv')), 'warmup': glob(os.path.join(data_directory, 'cmdstanpy', 'cmdstanpy_eight_schools_warmup-[1-4].csv'))}
        return files

    @pytest.fixture(scope='class')
    def data(self, filepaths):
        cmdstanpy = importorskip('cmdstanpy')
        CmdStanModel = cmdstanpy.CmdStanModel
        CmdStanMCMC = cmdstanpy.CmdStanMCMC
        RunSet = cmdstanpy.stanfit.RunSet
        CmdStanArgs = cmdstanpy.model.CmdStanArgs
        SamplerArgs = cmdstanpy.model.SamplerArgs

        class Data:
            args = CmdStanArgs('dummy.stan', 'dummy.exe', list(range(1, 5)), method_args=SamplerArgs(iter_sampling=100))
            runset_obj = RunSet(args)
            runset_obj._csv_files = filepaths['nowarmup']
            obj = CmdStanMCMC(runset_obj)
            obj._assemble_draws()
            args_warmup = CmdStanArgs('dummy.stan', 'dummy.exe', list(range(1, 5)), method_args=SamplerArgs(iter_sampling=100, iter_warmup=500, save_warmup=True))
            runset_obj_warmup = RunSet(args_warmup)
            runset_obj_warmup._csv_files = filepaths['warmup']
            obj_warmup = CmdStanMCMC(runset_obj_warmup)
            obj_warmup._assemble_draws()
            _model_code = 'model { real y; } generated quantities { int eta; int theta[N]; }'
            _tmp_dir = tempfile.TemporaryDirectory(prefix='arviz_tests_')
            _stan_file = os.path.join(_tmp_dir.name, 'stan_model_test.stan')
            with open(_stan_file, 'w', encoding='utf8') as f:
                f.write(_model_code)
            model = CmdStanModel(stan_file=_stan_file, compile=False)
        return Data

    def get_inference_data(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(posterior=data.obj, posterior_predictive='y_hat', predictions='y_hat', prior=data.obj, prior_predictive='y_hat', observed_data={'y': eight_schools_params['y']}, constant_data={'y': eight_schools_params['y']}, predictions_constant_data={'y': eight_schools_params['y']}, log_likelihood={'y': 'log_lik'}, coords={'school': np.arange(eight_schools_params['J'])}, dims={'y': ['school'], 'log_lik': ['school'], 'y_hat': ['school'], 'theta': ['school']})

    def get_inference_data2(self, data, eight_schools_params):
        """vars as lists."""
        return from_cmdstanpy(posterior=data.obj, posterior_predictive=['y_hat'], predictions=['y_hat', 'log_lik'], prior=data.obj, prior_predictive=['y_hat'], observed_data={'y': eight_schools_params['y']}, constant_data=eight_schools_params, predictions_constant_data=eight_schools_params, log_likelihood=['log_lik', 'y_hat'], coords={'school': np.arange(eight_schools_params['J']), 'log_lik_dim': np.arange(eight_schools_params['J'])}, dims={'eta': ['extra_dim', 'half school'], 'y': ['school'], 'y_hat': ['school'], 'theta': ['school'], 'log_lik': ['log_lik_dim']})

    def get_inference_data3(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(posterior=data.obj, posterior_predictive=['y_hat', 'log_lik'], prior=data.obj, prior_predictive=['y_hat', 'log_lik'], observed_data={'y': eight_schools_params['y']}, coords={'school': np.arange(eight_schools_params['J']), 'half school': ['a', 'b', 'c', 'd'], 'extra_dim': ['x', 'y']}, dims={'eta': ['extra_dim', 'half school'], 'y': ['school'], 'y_hat': ['school'], 'theta': ['school'], 'log_lik': ['log_lik_dim']}, dtypes=data.model)

    def get_inference_data4(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(posterior=data.obj, posterior_predictive=None, prior=data.obj, prior_predictive=None, log_likelihood=False, observed_data={'y': eight_schools_params['y']}, coords=None, dims=None, dtypes={'eta': int, 'theta': int})

    def get_inference_data5(self, data, eight_schools_params):
        """multiple vars as lists."""
        return from_cmdstanpy(posterior=data.obj, posterior_predictive=None, prior=data.obj, prior_predictive=None, log_likelihood='log_lik', observed_data={'y': eight_schools_params['y']}, coords=None, dims=None, dtypes=data.model.code())

    def get_inference_data_warmup_true_is_true(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(posterior=data.obj_warmup, posterior_predictive='y_hat', predictions='y_hat', prior=data.obj_warmup, prior_predictive='y_hat', observed_data={'y': eight_schools_params['y']}, constant_data={'y': eight_schools_params['y']}, predictions_constant_data={'y': eight_schools_params['y']}, log_likelihood='log_lik', coords={'school': np.arange(eight_schools_params['J'])}, dims={'eta': ['extra_dim', 'half school'], 'y': ['school'], 'log_lik': ['school'], 'y_hat': ['school'], 'theta': ['school']}, save_warmup=True)

    def get_inference_data_warmup_false_is_true(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(posterior=data.obj, posterior_predictive='y_hat', predictions='y_hat', prior=data.obj, prior_predictive='y_hat', observed_data={'y': eight_schools_params['y']}, constant_data={'y': eight_schools_params['y']}, predictions_constant_data={'y': eight_schools_params['y']}, log_likelihood='log_lik', coords={'school': np.arange(eight_schools_params['J'])}, dims={'eta': ['extra_dim', 'half school'], 'y': ['school'], 'log_lik': ['school'], 'y_hat': ['school'], 'theta': ['school']}, save_warmup=True)

    def get_inference_data_warmup_true_is_false(self, data, eight_schools_params):
        """vars as str."""
        return from_cmdstanpy(posterior=data.obj_warmup, posterior_predictive='y_hat', predictions='y_hat', prior=data.obj_warmup, prior_predictive='y_hat', observed_data={'y': eight_schools_params['y']}, constant_data={'y': eight_schools_params['y']}, predictions_constant_data={'y': eight_schools_params['y']}, log_likelihood='log_lik', coords={'school': np.arange(eight_schools_params['J'])}, dims={'eta': ['extra_dim', 'half school'], 'y': ['school'], 'log_lik': ['school'], 'y_hat': ['school'], 'theta': ['school']}, save_warmup=False)

    def test_sampler_stats(self, data, eight_schools_params):
        inference_data = self.get_inference_data(data, eight_schools_params)
        test_dict = {'sample_stats': ['lp', 'diverging']}
        fails = check_multiple_attrs(test_dict, inference_data)
        assert not fails
        assert len(inference_data.sample_stats.lp.shape) == 2

    def test_inference_data(self, data, eight_schools_params):
        inference_data1 = self.get_inference_data(data, eight_schools_params)
        inference_data2 = self.get_inference_data2(data, eight_schools_params)
        inference_data3 = self.get_inference_data3(data, eight_schools_params)
        inference_data4 = self.get_inference_data4(data, eight_schools_params)
        inference_data5 = self.get_inference_data5(data, eight_schools_params)
        test_dict = {'posterior': ['theta'], 'predictions': ['y_hat'], 'observed_data': ['y'], 'constant_data': ['y'], 'predictions_constant_data': ['y'], 'log_likelihood': ['y', '~log_lik'], 'prior': ['theta']}
        fails = check_multiple_attrs(test_dict, inference_data1)
        assert not fails
        test_dict = {'posterior_predictive': ['y_hat'], 'predictions': ['y_hat', 'log_lik'], 'observed_data': ['y'], 'sample_stats_prior': ['lp'], 'sample_stats': ['lp'], 'constant_data': list(eight_schools_params), 'predictions_constant_data': list(eight_schools_params), 'prior_predictive': ['y_hat'], 'log_likelihood': ['log_lik', 'y_hat']}
        fails = check_multiple_attrs(test_dict, inference_data2)
        assert not fails
        test_dict = {'posterior_predictive': ['y_hat'], 'observed_data': ['y'], 'sample_stats_prior': ['lp'], 'sample_stats': ['lp'], 'prior_predictive': ['y_hat'], 'log_likelihood': ['log_lik']}
        fails = check_multiple_attrs(test_dict, inference_data3)
        assert not fails
        assert inference_data3.posterior.eta.dtype.kind == 'i'
        assert inference_data3.posterior.theta.dtype.kind == 'i'
        test_dict = {'posterior': ['eta', 'mu', 'theta'], 'prior': ['theta'], '~log_likelihood': ['']}
        fails = check_multiple_attrs(test_dict, inference_data4)
        assert not fails
        assert len(inference_data4.posterior.theta.shape) == 3
        assert len(inference_data4.posterior.eta.shape) == 4
        assert len(inference_data4.posterior.mu.shape) == 2
        assert inference_data4.posterior.eta.dtype.kind == 'i'
        assert inference_data4.posterior.theta.dtype.kind == 'i'
        test_dict = {'posterior': ['eta', 'mu', 'theta'], 'prior': ['theta'], 'log_likelihood': ['log_lik']}
        fails = check_multiple_attrs(test_dict, inference_data5)
        assert inference_data5.posterior.eta.dtype.kind == 'i'
        assert inference_data5.posterior.theta.dtype.kind == 'i'

    def test_inference_data_warmup(self, data, eight_schools_params):
        inference_data_true_is_true = self.get_inference_data_warmup_true_is_true(data, eight_schools_params)
        inference_data_false_is_true = self.get_inference_data_warmup_false_is_true(data, eight_schools_params)
        inference_data_true_is_false = self.get_inference_data_warmup_true_is_false(data, eight_schools_params)
        inference_data_false_is_false = self.get_inference_data(data, eight_schools_params)
        test_dict = {'posterior': ['theta'], 'predictions': ['y_hat'], 'observed_data': ['y'], 'constant_data': ['y'], 'predictions_constant_data': ['y'], 'log_likelihood': ['log_lik'], 'prior': ['theta'], 'warmup_posterior': ['theta'], 'warmup_predictions': ['y_hat'], 'warmup_log_likelihood': ['log_lik'], 'warmup_prior': ['theta']}
        fails = check_multiple_attrs(test_dict, inference_data_true_is_true)
        assert not fails
        test_dict = {'posterior': ['theta'], 'predictions': ['y_hat'], 'observed_data': ['y'], 'constant_data': ['y'], 'predictions_constant_data': ['y'], 'log_likelihood': ['log_lik'], 'prior': ['theta'], '~warmup_posterior': [''], '~warmup_predictions': [''], '~warmup_log_likelihood': [''], '~warmup_prior': ['']}
        fails = check_multiple_attrs(test_dict, inference_data_false_is_true)
        assert not fails
        test_dict = {'posterior': ['theta'], 'predictions': ['y_hat'], 'observed_data': ['y'], 'constant_data': ['y'], 'predictions_constant_data': ['y'], 'log_likelihood': ['log_lik'], 'prior': ['theta'], '~warmup_posterior': [''], '~warmup_predictions': [''], '~warmup_log_likelihood': [''], '~warmup_prior': ['']}
        fails = check_multiple_attrs(test_dict, inference_data_true_is_false)
        assert not fails
        test_dict = {'posterior': ['theta'], 'predictions': ['y_hat'], 'observed_data': ['y'], 'constant_data': ['y'], 'predictions_constant_data': ['y'], 'log_likelihood': ['y'], 'prior': ['theta'], '~warmup_posterior': [''], '~warmup_predictions': [''], '~warmup_log_likelihood': [''], '~warmup_prior': ['']}
        fails = check_multiple_attrs(test_dict, inference_data_false_is_false)
        assert not fails