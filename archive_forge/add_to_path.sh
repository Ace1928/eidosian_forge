#!/usr/bin/env bash

DIR="$(pwd)"

echo "======================================="
echo "We are about to add the following path:"
echo "    $DIR"
echo
echo "Current PATH:"
echo "    $PATH"
echo
echo "If we proceed, your PATH will become:"
echo "    $PATH:$DIR"
echo

read -rp "Proceed with changes? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then

  # -- 1. Update ~/.bashrc if $DIR is not already present --
  if grep -F -q "$DIR" ~/.bashrc; then
    echo "=> Skipping update: $DIR is already present in your PATH."
  else
    echo "=> Adding $DIR to your personal PATH in ~/.bashrc..."
    echo "export PATH=\"\$PATH:$DIR\"" >> ~/.bashrc
  fi

  # -- 2. Reload ~/.bashrc so the change takes effect immediately --
  echo "=> Reloading ~/.bashrc to reflect changes in the current shell..."
  # shellcheck disable=SC1090
  source ~/.bashrc

  echo
  echo "=> Done!"
  echo "Your user PATH is now: $PATH"
  echo

else
  echo "=> Changes have been canceled by the user."
fi
