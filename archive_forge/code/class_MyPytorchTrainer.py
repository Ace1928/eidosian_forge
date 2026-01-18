import torch
from ray import train
from ray.train.trainer import BaseTrainer
import ray
class MyPytorchTrainer(BaseTrainer):

    def setup(self):
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def training_loop(self):
        dataset = self.datasets['train']
        loss_fn = torch.nn.MSELoss()
        for epoch_idx in range(10):
            loss = 0
            num_batches = 0
            for batch in dataset.iter_torch_batches(dtypes=torch.float):
                X, y = (torch.unsqueeze(batch['x'], 1), batch['y'])
                pred = self.model(X)
                batch_loss = loss_fn(pred, y)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                loss += batch_loss.item()
                num_batches += 1
            loss /= num_batches
            train.report({'loss': loss, 'epoch': epoch_idx})